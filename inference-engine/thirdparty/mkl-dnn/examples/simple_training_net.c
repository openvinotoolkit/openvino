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

// Required for posix_memalign
#define _POSIX_C_SOURCE 200112L

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkldnn.h"
#ifdef _WIN32
#include <malloc.h>
#endif

#define BATCH 32
#define IC 3
#define OC 96
#define CONV_IH 227
#define CONV_IW 227
#define CONV_OH 55
#define CONV_OW 55
#define CONV_STRIDE 4
#define CONV_PAD 0
#define POOL_OH 27
#define POOL_OW 27
#define POOL_STRIDE 2
#define POOL_PAD 0

#define CHECK(f)                                                               \
    do {                                                                       \
        mkldnn_status_t s = f;                                                 \
        if (s != mkldnn_success) {                                             \
            printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f,   \
                   s);                                                         \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_TRUE(expr)                                                       \
    do {                                                                       \
        int e_ = expr;                                                         \
        if (!e_) {                                                             \
            printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr);          \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

void *aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void *p;
    return !posix_memalign(&p, alignment, size) ? p : NULL;
#endif
}

#ifdef _WIN32
void _free(void *ptr) {
    _aligned_free(ptr);
}
#else
void _free(void *ptr) {
    free(ptr);
}
#endif

static size_t product(int *arr, size_t size)
{
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= arr[i];
    return prod;
}

static void init_net_data(float *data, uint32_t dim, const int *dims)
{
    if (dim == 1) {
        for (int i = 0; i < dims[0]; ++i) {
            data[i] = (float)(i % 1637);
        }
    } else if (dim == 4) {
        for (int in = 0; in < dims[0]; ++in) {
            for (int ic = 0; ic < dims[1]; ++ic) {
                for (int ih = 0; ih < dims[2]; ++ih) {
                    for (int iw = 0; iw < dims[3]; ++iw) {
                        int indx = in * dims[1] * dims[2] * dims[3]
                                   + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx] = (float)(indx % 1637);
                    }
                }
            }
        }
    }
}

static void init_data_memory(uint32_t dim, const int *dims,
                             mkldnn_memory_format_t user_fmt,
                             mkldnn_data_type_t data_type,
                             mkldnn_engine_t engine, float *data,
                             mkldnn_primitive_t *memory)
{
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    CHECK(mkldnn_memory_desc_init(&prim_md, dim, dims, data_type, user_fmt));
    CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
    CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));

    void *req = NULL;
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == NULL);
    CHECK(mkldnn_memory_set_data_handle(*memory, data));
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == data);
    CHECK(mkldnn_primitive_desc_destroy(user_pd));
}

mkldnn_status_t
prepare_reorder(mkldnn_primitive_t *user_memory,               /** in */
                const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
                int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
                mkldnn_primitive_t *prim_memory, mkldnn_primitive_t
                *reorder, /** out: reorder primitive created */
                float *buffer)
{
    const_mkldnn_primitive_desc_t user_memory_pd;
    mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
        CHECK(mkldnn_primitive_create(prim_memory, *prim_memory_pd, NULL,
                                      NULL));
        CHECK(mkldnn_memory_set_data_handle(*prim_memory, buffer));

        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
            CHECK(mkldnn_reorder_primitive_desc_create(
                    &reorder_pd, user_memory_pd, *prim_memory_pd));
            mkldnn_primitive_at_t inputs = { *user_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *prim_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                                          outputs));
        } else {
            CHECK(mkldnn_reorder_primitive_desc_create(
                    &reorder_pd, *prim_memory_pd, user_memory_pd));
            mkldnn_primitive_at_t inputs = { *prim_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *user_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                                          outputs));
        }
        CHECK(mkldnn_primitive_desc_destroy(reorder_pd));
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return mkldnn_success;
}

mkldnn_status_t simple_net()
{

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));

    int net_src_sizes[4] = { BATCH, IC, CONV_IH, CONV_IW };
    int net_dst_sizes[4] = { BATCH, OC, POOL_OH, POOL_OW };

    float *net_src =
        (float *)aligned_malloc(product(net_src_sizes,4)*sizeof(float), 64);
    float *net_dst =
        (float *)aligned_malloc(product(net_dst_sizes, 4)*sizeof(float), 64);

    init_net_data(net_src, 4, net_src_sizes);
    memset(net_dst, 0, product(net_dst_sizes, 4)*sizeof(float));

    /*----------------------------------------------------------------------*/
    /*----------------- Forward Stream -------------------------------------*/
    /* AlexNet: conv
     * {BATCH, IC, CONV_IH, CONV_IW} (x) {OC, IC, 11, 11} ->
     * {BATCH, OC, CONV_OH, CONV_OW}
     * strides: {CONV_STRIDE, CONV_STRIDE}
     */
    int *conv_user_src_sizes = net_src_sizes;
    int conv_user_weights_sizes[4] = { OC, IC, 11, 11 };
    int conv_bias_sizes[4] = { OC };
    int conv_user_dst_sizes[4] = { BATCH, OC, CONV_OH, CONV_OW };
    int conv_strides[2] = { CONV_STRIDE, CONV_STRIDE };
    int conv_padding[2] = { CONV_PAD, CONV_PAD };

    float *conv_src = net_src;
    float *conv_weights = (float *)aligned_malloc(
            product(conv_user_weights_sizes, 4) * sizeof(float), 64);
    float *conv_bias = (float *)aligned_malloc(
            product(conv_bias_sizes, 1) * sizeof(float), 64);

    init_net_data(conv_weights, 4, conv_user_weights_sizes);
    init_net_data(conv_bias, 1, conv_bias_sizes);

    /* create memory for user data */
    mkldnn_primitive_t conv_user_src_memory, conv_user_weights_memory,
            conv_user_bias_memory;
    init_data_memory(4, conv_user_src_sizes, mkldnn_nchw, mkldnn_f32, engine,
                     conv_src, &conv_user_src_memory);
    init_data_memory(4, conv_user_weights_sizes, mkldnn_oihw, mkldnn_f32,
            engine, conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, mkldnn_x, mkldnn_f32, engine,
            conv_bias, &conv_user_bias_memory);

    /* create data descriptors for convolution w/ no specified format */
    mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
            conv_dst_md;
    CHECK(mkldnn_memory_desc_init(
            &conv_src_md, 4, conv_user_src_sizes, mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_weights_md, 4, conv_user_weights_sizes,
            mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(
            &conv_bias_md, 1, conv_bias_sizes, mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_desc_init(
            &conv_dst_md, 4, conv_user_dst_sizes, mkldnn_f32, mkldnn_any));

    /* create a convolution */
    mkldnn_convolution_desc_t conv_any_desc;
    CHECK(mkldnn_convolution_forward_desc_init(
            &conv_any_desc, mkldnn_forward, mkldnn_convolution_direct,
            &conv_src_md, &conv_weights_md, &conv_bias_md, &conv_dst_md,
            conv_strides, conv_padding, conv_padding, mkldnn_padding_zero));

    mkldnn_primitive_desc_t conv_pd;
    CHECK(mkldnn_primitive_desc_create(&conv_pd, &conv_any_desc, engine, NULL));

    mkldnn_primitive_t conv_internal_src_memory, conv_internal_weights_memory,
            conv_internal_dst_memory;

    /* create memory for dst data, we don't need to reorder it to user data */
    const_mkldnn_primitive_desc_t conv_dst_pd
            = mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(
            &conv_internal_dst_memory, conv_dst_pd, NULL, NULL));
    size_t conv_dst_size = mkldnn_memory_primitive_desc_get_size(conv_dst_pd);
    float *conv_dst_buffer = (float *)aligned_malloc(conv_dst_size, 64);
    CHECK(mkldnn_memory_set_data_handle(
            conv_internal_dst_memory, conv_dst_buffer));

    /* create reorder primitives between user data and convolution srcs
     * if required */
    mkldnn_primitive_t conv_reorder_src, conv_reorder_weights;

    const_mkldnn_primitive_desc_t conv_src_pd
            = mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_src_pd, 0);
    size_t conv_src_size = mkldnn_memory_primitive_desc_get_size(conv_src_pd);
    float *conv_src_buffer = (float *)aligned_malloc(conv_src_size, 64);
    CHECK(prepare_reorder(&conv_user_src_memory, &conv_src_pd, 1,
        &conv_internal_src_memory, &conv_reorder_src, conv_src_buffer));

    const_mkldnn_primitive_desc_t conv_weights_pd
            = mkldnn_primitive_desc_query_pd(
                    conv_pd, mkldnn_query_weights_pd, 0);
    size_t conv_weights_size
            = mkldnn_memory_primitive_desc_get_size(conv_weights_pd);
    float *conv_weights_buffer = (float *)aligned_malloc(conv_weights_size, 64);
    CHECK(prepare_reorder(&conv_user_weights_memory, &conv_weights_pd, 1,
            &conv_internal_weights_memory, &conv_reorder_weights,
            conv_weights_buffer));

    mkldnn_primitive_t conv_src_memory = conv_internal_src_memory
                                                ? conv_internal_src_memory
                                                : conv_user_src_memory;
    mkldnn_primitive_t conv_weights_memory = conv_internal_weights_memory
                                                ? conv_internal_weights_memory
                                                : conv_user_weights_memory;

    mkldnn_primitive_at_t conv_srcs[]
            = { mkldnn_primitive_at(conv_src_memory, 0),
                mkldnn_primitive_at(conv_weights_memory, 0),
                mkldnn_primitive_at(conv_user_bias_memory, 0) };

    const_mkldnn_primitive_t conv_dsts[] = { conv_internal_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv;
    CHECK(mkldnn_primitive_create(&conv, conv_pd, conv_srcs, conv_dsts));

    /* AlexNet: relu
     * {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
     */
    float negative_slope = 1.0f;

    /* keep memory format of source same as the format of convolution
       * output in order to avoid reorder */
    const mkldnn_memory_desc_t *relu_src_md
            = mkldnn_primitive_desc_query_memory_d(conv_dst_pd);

    /* create a relu primitive descriptor */
    mkldnn_eltwise_desc_t relu_desc;
    CHECK(mkldnn_eltwise_forward_desc_init(&relu_desc, mkldnn_forward,
                mkldnn_eltwise_relu, relu_src_md, negative_slope, 0));

    mkldnn_primitive_desc_t relu_pd;
    CHECK(mkldnn_primitive_desc_create(&relu_pd, &relu_desc, engine, NULL));

    /* create relu dst memory primitive */
    mkldnn_primitive_t relu_dst_memory;
    const_mkldnn_primitive_desc_t relu_dst_pd
            = mkldnn_primitive_desc_query_pd(relu_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(&relu_dst_memory, relu_dst_pd, NULL, NULL));
    size_t relu_dst_size = mkldnn_memory_primitive_desc_get_size(relu_dst_pd);
    float *relu_dst_buffer = (float *)aligned_malloc(relu_dst_size, 64);
    CHECK(mkldnn_memory_set_data_handle(relu_dst_memory, relu_dst_buffer));

    /* finally create a relu primitive */
    mkldnn_primitive_t relu;
    mkldnn_primitive_at_t relu_srcs = { conv_internal_dst_memory, 0 };
    const_mkldnn_primitive_t relu_dsts[] = { relu_dst_memory };

    CHECK(mkldnn_primitive_create(&relu, relu_pd, &relu_srcs, relu_dsts));

    /* AlexNet: lrn
     * {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
     * local size: 5
     * alpha: 0.0001
     * beta: 0.75
     * k: 1.0
     */
    uint32_t local_size = 5;
    float alpha = 0.0001f;
    float beta = 0.75f;
    float k = 1.0f;

    /* create lrn src memory descriptor using dst memory descriptor
     *  from previous primitive */
    const mkldnn_memory_desc_t *lrn_src_md
            = mkldnn_primitive_desc_query_memory_d(relu_dst_pd);

    /* create a lrn primitive descriptor */
    mkldnn_lrn_desc_t lrn_desc;
    CHECK(mkldnn_lrn_forward_desc_init(&lrn_desc, mkldnn_forward,
                                       mkldnn_lrn_across_channels, lrn_src_md,
                                       local_size, alpha, beta, k));

    mkldnn_primitive_desc_t lrn_pd;
    CHECK(mkldnn_primitive_desc_create(&lrn_pd, &lrn_desc, engine, NULL));

    /* create primitives for lrn dst and workspace memory */
    mkldnn_primitive_t lrn_dst_memory, lrn_workspace_memory;

    const_mkldnn_primitive_desc_t lrn_dst_pd
            = mkldnn_primitive_desc_query_pd(lrn_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(&lrn_dst_memory, lrn_dst_pd, NULL, NULL));
    size_t lrn_dst_size = mkldnn_memory_primitive_desc_get_size(lrn_dst_pd);
    float *lrn_dst_buffer = (float *)aligned_malloc(lrn_dst_size, 64);
    CHECK(mkldnn_memory_set_data_handle(lrn_dst_memory, lrn_dst_buffer));

    /* create workspace only in training and only for forward primitive*/
    /* query lrn_pd for workspace, this memory will be shared with forward lrn*/
    const_mkldnn_primitive_desc_t lrn_workspace_pd
            = mkldnn_primitive_desc_query_pd(lrn_pd, mkldnn_query_workspace_pd,
                                             0);
    CHECK(mkldnn_primitive_create(&lrn_workspace_memory, lrn_workspace_pd, NULL,
                                  NULL));
    size_t lrn_workspace_size =
        mkldnn_memory_primitive_desc_get_size(lrn_workspace_pd);
    float *lrn_workspace_buffer =
        (float*)aligned_malloc(lrn_workspace_size, 64);
    memset(lrn_workspace_buffer, 0, lrn_workspace_size);
    CHECK(mkldnn_memory_set_data_handle(lrn_workspace_memory,
                                        lrn_workspace_buffer));

    mkldnn_primitive_at_t lrn_srcs = { relu_dst_memory, 0 };

    const_mkldnn_primitive_t lrn_dsts[]
            = { lrn_dst_memory, lrn_workspace_memory };

    /* finally create a lrn primitive */
    mkldnn_primitive_t lrn;
    CHECK(mkldnn_primitive_create(&lrn, lrn_pd, &lrn_srcs, lrn_dsts));

    /* AlexNet: pool
     * {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, POOL_OH, POOL_OW}
     * kernel: {3, 3}
     * strides: {POOL_STRIDE, POOL_STRIDE}
     */
    int32_t *pool_dst_sizes = net_dst_sizes;
    int32_t pool_kernel[2] = { 3, 3 };
    int32_t pool_strides[2] = { POOL_STRIDE, POOL_STRIDE };
    int32_t pool_padding[2] = { POOL_PAD, POOL_PAD };

    /* create pooling src memory descriptor using dst descriptor
     *  from previous primitive */
    const mkldnn_memory_desc_t *pool_src_md
            = mkldnn_primitive_desc_query_memory_d(lrn_dst_pd);

    /* create descriptors for dst pooling data */
    mkldnn_memory_desc_t pool_dst_md;
    CHECK(mkldnn_memory_desc_init(&pool_dst_md, 4, pool_dst_sizes, mkldnn_f32,
                                  mkldnn_any));

    /* create memory for user dst data */
    mkldnn_primitive_t pool_user_dst_memory;
    init_data_memory(4, pool_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
                     net_dst, &pool_user_dst_memory);

    /* create a pooling primitive descriptor */
    mkldnn_pooling_desc_t pool_desc;
    CHECK(mkldnn_pooling_forward_desc_init(
            &pool_desc, mkldnn_forward, mkldnn_pooling_max, pool_src_md,
            &pool_dst_md, pool_strides, pool_kernel, pool_padding, pool_padding,
            mkldnn_padding_zero));

    mkldnn_primitive_desc_t pool_pd;
    CHECK(mkldnn_primitive_desc_create(&pool_pd, &pool_desc, engine, NULL));

    /* create memory for workspace */
    mkldnn_primitive_t pool_workspace_memory;
    const_mkldnn_primitive_desc_t pool_workspace_pd
            = mkldnn_primitive_desc_query_pd(pool_pd, mkldnn_query_workspace_pd,
                                             0);
    CHECK(mkldnn_primitive_create(&pool_workspace_memory, pool_workspace_pd,
                                  NULL, NULL));
    size_t pool_workspace_size =
        mkldnn_memory_primitive_desc_get_size(pool_workspace_pd);
    float *pool_workspace_buffer =
        (float*)aligned_malloc(pool_workspace_size, 64);
    memset(pool_workspace_buffer, 0, pool_workspace_size);
    CHECK(mkldnn_memory_set_data_handle(pool_workspace_memory,
                                        pool_workspace_buffer));

    mkldnn_primitive_t pool_dst_memory;

    /* create reorder primitives between pooling dsts and user format dst
     * if required */
    mkldnn_primitive_t pool_reorder_dst, pool_internal_dst_memory;
    const_mkldnn_primitive_desc_t pool_dst_pd
            = mkldnn_primitive_desc_query_pd(pool_pd, mkldnn_query_dst_pd, 0);
    size_t pool_dst_size = mkldnn_memory_primitive_desc_get_size(pool_dst_pd);
    float *pool_dst_buffer = (float *)aligned_malloc(pool_dst_size, 64);
    CHECK(prepare_reorder(&pool_user_dst_memory, &pool_dst_pd, 0,
                          &pool_internal_dst_memory, &pool_reorder_dst,
                          pool_dst_buffer));

    mkldnn_primitive_at_t pool_srcs = { lrn_dst_memory, 0 };

    pool_dst_memory = pool_internal_dst_memory ? pool_internal_dst_memory
                                               : pool_user_dst_memory;

    const_mkldnn_primitive_t pool_dsts[]
            = { pool_dst_memory, pool_workspace_memory };

    /* finally create a pooling primitive */
    mkldnn_primitive_t pool;
    CHECK(mkldnn_primitive_create(&pool, pool_pd, &pool_srcs, pool_dsts));

    /* build a simple net */
    uint32_t n_fwd = 0;
    mkldnn_primitive_t net_fwd[10];

    if (conv_reorder_src)
        net_fwd[n_fwd++] = conv_reorder_src;
    if (conv_reorder_weights)
        net_fwd[n_fwd++] = conv_reorder_weights;
    net_fwd[n_fwd++] = conv;
    net_fwd[n_fwd++] = relu;
    net_fwd[n_fwd++] = lrn;
    net_fwd[n_fwd++] = pool;
    if (pool_reorder_dst)
        net_fwd[n_fwd++] = pool_reorder_dst;

    void *net_output = NULL; // output from forward stream:

    /*----------------------------------------------------------------------*/
    /*----------------- Backward Stream -------------------------------------*/
    /* ... user diff_data ...*/
    float *net_diff_dst = (float *)aligned_malloc(
        product(pool_dst_sizes, 4) * sizeof(float), 64);

    init_net_data(net_diff_dst, 4, pool_dst_sizes);

    /* create memory primitives for user diff dst data*/
    mkldnn_primitive_t pool_user_diff_dst_memory;
    init_data_memory(4, pool_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
                     net_diff_dst, &pool_user_diff_dst_memory);

    /* Pooling Backward */
    /* pooling diff src memory descriptor */
    const mkldnn_memory_desc_t *pool_diff_src_md
            = mkldnn_primitive_desc_query_memory_d(lrn_dst_pd);

    /* pooling diff dst memory descriptor */
    const mkldnn_memory_desc_t *pool_diff_dst_md
            = mkldnn_primitive_desc_query_memory_d(pool_dst_pd);

    /* create backward pooling descriptor */
    mkldnn_pooling_desc_t pool_bwd_desc;
    CHECK(mkldnn_pooling_backward_desc_init(
            &pool_bwd_desc, mkldnn_pooling_max, pool_diff_src_md,
            pool_diff_dst_md, pool_strides, pool_kernel, pool_padding,
            pool_padding, mkldnn_padding_zero));

    /* backward primitive descriptor needs to hint forward descriptor*/
    mkldnn_primitive_desc_t pool_bwd_pd;
    CHECK(mkldnn_primitive_desc_create(&pool_bwd_pd, &pool_bwd_desc, engine,
                                       pool_pd));

    /* create reorder primitive between user diff dst and pool diff dst
     * if required*/
    mkldnn_primitive_t pool_diff_dst_memory;
    mkldnn_primitive_t pool_reorder_diff_dst, pool_internal_diff_dst_memory;
    const_mkldnn_primitive_desc_t pool_diff_dst_pd
            = mkldnn_primitive_desc_query_pd(pool_bwd_pd,
                                             mkldnn_query_diff_dst_pd, 0);
    size_t pool_diff_dst_size
        = mkldnn_memory_primitive_desc_get_size(pool_diff_dst_pd);
    float *pool_diff_dst_buffer
        = (float *)aligned_malloc(pool_diff_dst_size, 64);
    CHECK(prepare_reorder(&pool_user_diff_dst_memory, &pool_diff_dst_pd, 1,
                          &pool_internal_diff_dst_memory,
                          &pool_reorder_diff_dst, pool_diff_dst_buffer));

    pool_diff_dst_memory = pool_internal_diff_dst_memory
                                   ? pool_internal_diff_dst_memory
                                   : pool_user_diff_dst_memory;

    /* create memory primitive for pool diff src data */
    mkldnn_primitive_t pool_diff_src_memory;
    const_mkldnn_primitive_desc_t pool_diff_src_pd
            = mkldnn_primitive_desc_query_pd(pool_bwd_pd,
                                             mkldnn_query_diff_src_pd, 0);
    size_t pool_diff_src_size
            = mkldnn_memory_primitive_desc_get_size(pool_diff_src_pd);
    float *pool_diff_src_buffer
            = (float *)aligned_malloc(pool_diff_src_size, 64);
    CHECK(mkldnn_primitive_create(
            &pool_diff_src_memory, pool_diff_src_pd, NULL, NULL));
    CHECK(mkldnn_memory_set_data_handle(pool_diff_src_memory,
                                        pool_diff_src_buffer));

    mkldnn_primitive_at_t pool_diff_dsts[]
            = { mkldnn_primitive_at(pool_diff_dst_memory, 0),
                mkldnn_primitive_at(pool_workspace_memory, 0) };

    const_mkldnn_primitive_t pool_diff_srcs[] = { pool_diff_src_memory };

    /* finally create backward pooling primitive */
    mkldnn_primitive_t pool_bwd;
    CHECK(mkldnn_primitive_create(&pool_bwd, pool_bwd_pd, pool_diff_dsts,
                                  pool_diff_srcs));

    /* Backward lrn */
    const mkldnn_memory_desc_t *lrn_diff_dst_md
            = mkldnn_primitive_desc_query_memory_d(pool_diff_src_pd);

    /* create backward lrn descriptor */
    mkldnn_lrn_desc_t lrn_bwd_desc;
    CHECK(mkldnn_lrn_backward_desc_init(
            &lrn_bwd_desc, mkldnn_lrn_across_channels, lrn_src_md,
            lrn_diff_dst_md, local_size, alpha, beta, k));

    mkldnn_primitive_desc_t lrn_bwd_pd;
    CHECK(mkldnn_primitive_desc_create(&lrn_bwd_pd, &lrn_bwd_desc, engine,
                                       lrn_pd));

    /* create memory primitives for lrn diff src */
    mkldnn_primitive_t lrn_diff_src_memory;
    const_mkldnn_primitive_desc_t lrn_diff_src_pd
            = mkldnn_primitive_desc_query_pd(lrn_bwd_pd,
                                             mkldnn_query_diff_src_pd, 0);
    size_t lrn_diff_src_size
            = mkldnn_memory_primitive_desc_get_size(lrn_diff_src_pd);
    float *lrn_diff_src_buffer = (float *)aligned_malloc(lrn_diff_src_size, 64);
    CHECK(mkldnn_primitive_create(&lrn_diff_src_memory, lrn_diff_src_pd, NULL,
                                  NULL));
    CHECK(mkldnn_memory_set_data_handle(lrn_diff_src_memory,
                                        lrn_diff_src_buffer));

    mkldnn_primitive_at_t lrn_diff_dsts[]
            = { mkldnn_primitive_at(relu_dst_memory,
                                    0), // lrn_bwd requires src as first input
                mkldnn_primitive_at(pool_diff_src_memory, 0),
                mkldnn_primitive_at(lrn_workspace_memory, 0) };

    const_mkldnn_primitive_t lrn_diff_srcs[] = { lrn_diff_src_memory };

    /* finally create backward lrn primitive */
    mkldnn_primitive_t lrn_bwd;
    CHECK(mkldnn_primitive_create(&lrn_bwd, lrn_bwd_pd, lrn_diff_dsts,
             lrn_diff_srcs));

    /* Backward relu */
    const mkldnn_memory_desc_t *relu_diff_dst_md
            = mkldnn_primitive_desc_query_memory_d(lrn_diff_src_pd);

    /* create backward relu descriptor */
    mkldnn_eltwise_desc_t relu_bwd_desc;
    CHECK(mkldnn_eltwise_backward_desc_init(&relu_bwd_desc,
                mkldnn_eltwise_relu, relu_diff_dst_md, relu_src_md,
                negative_slope, 0));

    mkldnn_primitive_desc_t relu_bwd_pd;
    CHECK(mkldnn_primitive_desc_create(&relu_bwd_pd, &relu_bwd_desc, engine,
                                       relu_pd));

    /* create memory primitives for relu diff src */
    mkldnn_primitive_t relu_diff_src_memory;
    const_mkldnn_primitive_desc_t relu_diff_src_pd
            = mkldnn_primitive_desc_query_pd(relu_bwd_pd,
                                             mkldnn_query_diff_src_pd, 0);
    size_t relu_diff_src_size
            = mkldnn_memory_primitive_desc_get_size(relu_diff_src_pd);
    float *relu_diff_src_buffer
            = (float *)aligned_malloc(relu_diff_src_size, 64);

    CHECK(mkldnn_primitive_create(&relu_diff_src_memory, relu_diff_src_pd, NULL,
                                  NULL));
    CHECK(mkldnn_memory_set_data_handle(relu_diff_src_memory,
                                        relu_diff_src_buffer));

    mkldnn_primitive_at_t relu_diff_dsts[]
            = { mkldnn_primitive_at(conv_internal_dst_memory, 0),
                mkldnn_primitive_at(lrn_diff_src_memory, 0) };

    const_mkldnn_primitive_t relu_diff_srcs[] = { relu_diff_src_memory };

    /* finally create backward relu primitive */
    mkldnn_primitive_t relu_bwd;
    CHECK(mkldnn_primitive_create(&relu_bwd, relu_pd, relu_diff_dsts,
                                  relu_diff_srcs));

    /* Backward convolution with respect to weights */
    float *conv_diff_bias_buffer = (float *)aligned_malloc(
            product(conv_bias_sizes, 1) * sizeof(float), 64);
    float *conv_user_diff_weights_buffer = (float *)aligned_malloc(
            product(conv_user_weights_sizes, 4) * sizeof(float), 64);

    /* initialize memory for diff weights in user format */
    mkldnn_primitive_t conv_user_diff_weights_memory;
    init_data_memory(4, conv_user_weights_sizes, mkldnn_nchw, mkldnn_f32,
            engine, conv_user_diff_weights_buffer,
            &conv_user_diff_weights_memory);

    /* memory descriptors should be in format `any` to allow backward
     * convolution for
     * weights to chose the format it prefers for best performance */
    mkldnn_memory_desc_t conv_diff_src_md, conv_diff_weights_md,
            conv_diff_bias_md, conv_diff_dst_md;
    CHECK(mkldnn_memory_desc_init(
            &conv_diff_src_md, 4, conv_user_src_sizes, mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_diff_weights_md, 4,
            conv_user_weights_sizes, mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(
            &conv_diff_bias_md, 1, conv_bias_sizes, mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_desc_init(
            &conv_diff_dst_md, 4, conv_user_dst_sizes, mkldnn_f32, mkldnn_any));

    /* create backward convolution descriptor */
    mkldnn_convolution_desc_t conv_bwd_weights_desc;
    CHECK(mkldnn_convolution_backward_weights_desc_init(&conv_bwd_weights_desc,
            mkldnn_convolution_direct, &conv_diff_src_md, &conv_diff_weights_md,
            &conv_diff_bias_md, &conv_diff_dst_md, conv_strides, conv_padding,
            conv_padding, mkldnn_padding_zero));

    mkldnn_primitive_desc_t conv_bwd_weights_pd;
    CHECK(mkldnn_primitive_desc_create(
            &conv_bwd_weights_pd, &conv_bwd_weights_desc, engine, conv_pd));

    /* for best performance convolution backward might chose
     * different memory format for src and diff_dst
     * than the memory formats preferred by forward convolution
     * for src and dst respectively */
    /* create reorder primitives for src from forward convolution to the
     * format chosen by backward convolution */
    mkldnn_primitive_t conv_bwd_reorder_src, conv_bwd_internal_src_memory;
    const_mkldnn_primitive_desc_t conv_diff_src_pd
            = mkldnn_primitive_desc_query_pd(conv_bwd_weights_pd,
                                             mkldnn_query_src_pd, 0);
    size_t conv_diff_src_size
            = mkldnn_memory_primitive_desc_get_size(conv_diff_src_pd);
    float *conv_diff_src_buffer
            = (float *)aligned_malloc(conv_diff_src_size, 64);
    CHECK(prepare_reorder(&conv_src_memory, &conv_diff_src_pd, 1,
            &conv_bwd_internal_src_memory, &conv_bwd_reorder_src,
            conv_diff_src_buffer));

    mkldnn_primitive_t conv_diff_src_memory
            = conv_bwd_internal_src_memory ? conv_bwd_internal_src_memory
                                           : conv_src_memory;

    /* create reorder primitives for diff_dst between diff_src from relu_bwd
     * and format preferred by conv_diff_weights */
    mkldnn_primitive_t conv_reorder_diff_dst, conv_internal_diff_dst_memory;
    const_mkldnn_primitive_desc_t conv_diff_dst_pd
            = mkldnn_primitive_desc_query_pd(conv_bwd_weights_pd,
                                             mkldnn_query_diff_dst_pd, 0);
    size_t conv_diff_dst_size
            = mkldnn_memory_primitive_desc_get_size(conv_diff_dst_pd);
    float *conv_diff_dst_buffer
            = (float *)aligned_malloc(conv_diff_dst_size, 64);

    CHECK(prepare_reorder(&relu_diff_src_memory, &conv_diff_dst_pd, 1,
                          &conv_internal_diff_dst_memory,
                          &conv_reorder_diff_dst, conv_diff_dst_buffer));

    mkldnn_primitive_t conv_diff_dst_memory
            = conv_internal_diff_dst_memory ? conv_internal_diff_dst_memory
                                            : relu_diff_src_memory;

    /* create reorder primitives for conv diff weights memory */
    mkldnn_primitive_t conv_reorder_diff_weights,
            conv_internal_diff_weights_memory;
    const_mkldnn_primitive_desc_t conv_diff_weights_pd
            = mkldnn_primitive_desc_query_pd(conv_bwd_weights_pd,
                                             mkldnn_query_diff_weights_pd, 0);
    size_t conv_diff_weights_size
            = mkldnn_memory_primitive_desc_get_size(conv_diff_weights_pd);
    float *conv_diff_weights_buffer
            = (float *)aligned_malloc(conv_diff_weights_size, 64);
    CHECK(prepare_reorder(&conv_user_diff_weights_memory, &conv_diff_weights_pd,
                          0, &conv_internal_diff_weights_memory,
                          &conv_reorder_diff_weights,
                          conv_diff_weights_buffer));

    mkldnn_primitive_t conv_diff_weights_memory
            = conv_internal_diff_weights_memory
                      ? conv_internal_diff_weights_memory
                      : conv_user_diff_weights_memory;

    /* create memory primitive for diff bias memory */
    mkldnn_primitive_t conv_diff_bias_memory;
    mkldnn_primitive_desc_t conv_diff_bias_pd;
    CHECK(mkldnn_memory_primitive_desc_create(&conv_diff_bias_pd,
                                              &conv_diff_bias_md, engine));
    CHECK(mkldnn_primitive_create(&conv_diff_bias_memory, conv_diff_bias_pd,
                                  NULL, NULL));
    CHECK(mkldnn_memory_set_data_handle(conv_diff_bias_memory,
                                        conv_diff_bias_buffer));

    mkldnn_primitive_at_t conv_diff_dsts[]
            = { mkldnn_primitive_at(conv_diff_src_memory, 0),
                mkldnn_primitive_at(conv_diff_dst_memory, 0) };

    const_mkldnn_primitive_t conv_diff_weights[]
            = { conv_diff_weights_memory, conv_diff_bias_memory };

    /* finally created backward convolution weights primitive */
    mkldnn_primitive_t conv_bwd_weights;
    CHECK(mkldnn_primitive_create(&conv_bwd_weights, conv_bwd_weights_pd,
                                  conv_diff_dsts, conv_diff_weights));

    /* build backward stream */
    uint32_t n_bwd = 0;
    mkldnn_primitive_t net_bwd[10];

    if (pool_reorder_diff_dst)
        net_bwd[n_bwd++] = pool_reorder_diff_dst;
    net_bwd[n_bwd++] = pool_bwd;
    net_bwd[n_bwd++] = lrn_bwd;
    net_bwd[n_bwd++] = relu_bwd;
    if (conv_bwd_reorder_src)
        net_bwd[n_bwd++] = conv_bwd_reorder_src;
    if (conv_reorder_diff_dst)
        net_bwd[n_bwd++] = conv_reorder_diff_dst;
    net_bwd[n_bwd++] = conv_bwd_weights;
    if (conv_reorder_diff_weights)
        net_bwd[n_bwd++] = conv_reorder_diff_weights;

    // output from backward stream
    void *net_diff_weights = NULL;
    void *net_diff_bias = NULL;

    int n_iter = 10; //number of iterations for training.
    /* Execute the net */
    for (int i = 0; i < n_iter; i++) {
        mkldnn_stream_t stream_fwd;
        CHECK(mkldnn_stream_create(&stream_fwd, mkldnn_eager));
        CHECK(mkldnn_stream_submit(stream_fwd, n_fwd, net_fwd, NULL));
        CHECK(mkldnn_stream_wait(stream_fwd, n_fwd, NULL));
        CHECK(mkldnn_stream_destroy(stream_fwd));

        /* Update net_diff_dst */
        CHECK(mkldnn_memory_get_data_handle(pool_user_dst_memory, &net_output));
        /*...user updates net_diff_dst using net_output...*/
        // some user defined func update_diff_dst(net_diff_dst, net_output)

        /* Backward pass */
        mkldnn_stream_t stream_bwd;
        CHECK(mkldnn_stream_create(&stream_bwd, mkldnn_eager));
        CHECK(mkldnn_stream_submit(stream_bwd, n_bwd, net_bwd, NULL));
        CHECK(mkldnn_stream_wait(stream_bwd, n_bwd, NULL));
        CHECK(mkldnn_stream_destroy(stream_bwd));

        /*... update weights ... */
        CHECK(mkldnn_memory_get_data_handle(conv_user_diff_weights_memory,
                                            &net_diff_weights));
        CHECK(mkldnn_memory_get_data_handle(conv_diff_bias_memory,
                                            &net_diff_bias));
        /* ...user updates weights and bias using diff weights and bias...*/
        // some user defined func update_weights(conv_user_weights_memory,
        // conv_bias_memory,
        //      net_diff_weights, net_diff_bias);
    }

    /* Cleanup forward */
    CHECK(mkldnn_primitive_desc_destroy(pool_pd));
    CHECK(mkldnn_primitive_desc_destroy(lrn_pd));
    CHECK(mkldnn_primitive_desc_destroy(relu_pd));
    CHECK(mkldnn_primitive_desc_destroy(conv_pd));

    _free(net_src);
    _free(net_dst);

    mkldnn_primitive_destroy(conv_user_src_memory);
    mkldnn_primitive_destroy(conv_user_weights_memory);
    mkldnn_primitive_destroy(conv_user_bias_memory);
    mkldnn_primitive_destroy(conv_internal_src_memory);
    mkldnn_primitive_destroy(conv_internal_weights_memory);
    mkldnn_primitive_destroy(conv_internal_dst_memory);
    mkldnn_primitive_destroy(conv_reorder_src);
    mkldnn_primitive_destroy(conv_reorder_weights);
    mkldnn_primitive_destroy(conv);

    _free(conv_weights);
    _free(conv_bias);

    _free(conv_src_buffer);
    _free(conv_weights_buffer);
    _free(conv_dst_buffer);

    mkldnn_primitive_destroy(relu_dst_memory);
    mkldnn_primitive_destroy(relu);

    _free(relu_dst_buffer);

    mkldnn_primitive_destroy(lrn_workspace_memory);
    mkldnn_primitive_destroy(lrn_dst_memory);
    mkldnn_primitive_destroy(lrn);

    _free(lrn_workspace_buffer);
    _free(lrn_dst_buffer);

    mkldnn_primitive_destroy(pool_user_dst_memory);
    mkldnn_primitive_destroy(pool_internal_dst_memory);
    mkldnn_primitive_destroy(pool_workspace_memory);
    mkldnn_primitive_destroy(pool_reorder_dst);
    mkldnn_primitive_destroy(pool);

    _free(pool_dst_buffer);
    _free(pool_workspace_buffer);

    /* Cleanup backward */
    CHECK(mkldnn_primitive_desc_destroy(pool_bwd_pd));
    CHECK(mkldnn_primitive_desc_destroy(lrn_bwd_pd));
    CHECK(mkldnn_primitive_desc_destroy(relu_bwd_pd));
    CHECK(mkldnn_primitive_desc_destroy(conv_diff_bias_pd));
    CHECK(mkldnn_primitive_desc_destroy(conv_bwd_weights_pd));

    mkldnn_primitive_destroy(pool_user_diff_dst_memory);
    mkldnn_primitive_destroy(pool_diff_src_memory);
    mkldnn_primitive_destroy(pool_internal_diff_dst_memory);
    mkldnn_primitive_destroy(pool_reorder_diff_dst);
    mkldnn_primitive_destroy(pool_bwd);

    _free(net_diff_dst);
    _free(pool_diff_dst_buffer);
    _free(pool_diff_src_buffer);

    mkldnn_primitive_destroy(lrn_diff_src_memory);
    mkldnn_primitive_destroy(lrn_bwd);

    _free(lrn_diff_src_buffer);

    mkldnn_primitive_destroy(relu_diff_src_memory);
    mkldnn_primitive_destroy(relu_bwd);

    _free(relu_diff_src_buffer);

    mkldnn_primitive_destroy(conv_user_diff_weights_memory);
    mkldnn_primitive_destroy(conv_diff_bias_memory);
    mkldnn_primitive_destroy(conv_bwd_internal_src_memory);
    mkldnn_primitive_destroy(conv_bwd_reorder_src);
    mkldnn_primitive_destroy(conv_internal_diff_dst_memory);
    mkldnn_primitive_destroy(conv_reorder_diff_dst);
    mkldnn_primitive_destroy(conv_internal_diff_weights_memory);
    mkldnn_primitive_destroy(conv_reorder_diff_weights);
    mkldnn_primitive_destroy(conv_bwd_weights);

    _free(conv_diff_weights_buffer);
    _free(conv_diff_bias_buffer);
    _free(conv_user_diff_weights_buffer);
    _free(conv_diff_src_buffer);
    _free(conv_diff_dst_buffer);

    mkldnn_engine_destroy(engine);

    return mkldnn_success;
}

int main(int argc, char **argv)
{
    mkldnn_status_t result = simple_net();
    printf("%s\n", (result == mkldnn_success) ? "passed" : "failed");
    return result;
}
