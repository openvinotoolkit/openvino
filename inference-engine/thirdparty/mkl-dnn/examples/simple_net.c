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
#include "mkldnn.h"
#ifdef _WIN32
#include <malloc.h>
#endif

#define BATCH 8

#define CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

#define CHECK_TRUE(expr) do { \
    int e_ = expr; \
    if (!e_) { \
        printf("[%s:%d] %s failed\n", __FILE__, __LINE__, #expr); \
        exit(2); \
    } \
} while(0)

void *aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#elif defined(_SX)
    return malloc(size);
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

static size_t product(int *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

static void init_data_memory(uint32_t dim, const int *dims,
        mkldnn_memory_format_t user_fmt, mkldnn_data_type_t mkldnn_f32,
        mkldnn_engine_t engine, float *data, mkldnn_primitive_t *memory)
{
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    CHECK(mkldnn_memory_desc_init(&prim_md, dim, dims, mkldnn_f32, user_fmt));
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

mkldnn_status_t prepare_reorder(
        mkldnn_primitive_t *user_memory, /** in */
        const_mkldnn_primitive_desc_t *prim_memory_pd, /** in */
        int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
        mkldnn_primitive_t *prim_memory, /** out: memory primitive created */
        mkldnn_primitive_t *reorder, /** out: reorder primitive created */
        float *buffer)
{
    const_mkldnn_primitive_desc_t user_memory_pd;
    mkldnn_primitive_get_primitive_desc(*user_memory, &user_memory_pd);

    if (!mkldnn_memory_primitive_desc_equal(user_memory_pd, *prim_memory_pd)) {
        /* memory_create(&p, m, NULL) means allocate memory */
        CHECK(mkldnn_primitive_create(prim_memory, *prim_memory_pd,
                NULL, NULL));
        mkldnn_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                        user_memory_pd, *prim_memory_pd));
            mkldnn_primitive_at_t inputs = { *user_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *prim_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                        outputs));
        } else {
            CHECK(mkldnn_reorder_primitive_desc_create(&reorder_pd,
                        *prim_memory_pd, user_memory_pd));
            mkldnn_primitive_at_t inputs = { *prim_memory, 0 };
            const_mkldnn_primitive_t outputs[] = { *user_memory };
            CHECK(mkldnn_primitive_create(reorder, reorder_pd, &inputs,
                        outputs));
        }
        CHECK(mkldnn_memory_set_data_handle(*prim_memory, buffer));
        CHECK(mkldnn_primitive_desc_destroy(reorder_pd));
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return mkldnn_success;
}

mkldnn_status_t simple_net(){

    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0 /* idx */));

    float *net_src = (float*)aligned_malloc(BATCH*3*227*227*sizeof(float), 64);
    float *net_dst = (float*)aligned_malloc(BATCH*96*27*27*sizeof(float), 64);

    /* AlexNet: conv
     * {BATCH, 3, 227, 227} (x) {96, 3, 11, 11} -> {BATCH, 96, 55, 55}
     * strides: {4, 4}
     */
    int conv_src_sizes[4] = {BATCH, 3, 227, 227};
    int conv_weights_sizes[4] = {96, 3, 11, 11};
    int conv_bias_sizes[4] = {96};
    int conv_dst_sizes[4] = {BATCH, 96, 55, 55};
    int conv_strides[2] = {4, 4};
    int conv_padding[2] = {0, 0};

    float *conv_src = net_src;
    float *conv_weights =
       (float*)aligned_malloc(product(conv_weights_sizes, 4)*sizeof(float), 64);
    float *conv_bias =
       (float*)aligned_malloc(product(conv_bias_sizes, 1)*sizeof(float), 64);

    /* create memory for user data */
    mkldnn_primitive_t conv_user_src_memory, conv_user_weights_memory,
        conv_user_bias_memory;
    init_data_memory(4, conv_src_sizes, mkldnn_nchw, mkldnn_f32, engine,
        conv_src, &conv_user_src_memory);
    init_data_memory(4, conv_weights_sizes, mkldnn_oihw, mkldnn_f32, engine,
        conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, mkldnn_x, mkldnn_f32, engine,
        conv_bias, &conv_user_bias_memory);

    /* create data descriptors for convolution w/ no specified format */

    mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md;
    CHECK(mkldnn_memory_desc_init(&conv_src_md, 4, conv_src_sizes,
        mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_weights_md, 4, conv_weights_sizes,
        mkldnn_f32, mkldnn_any));
    CHECK(mkldnn_memory_desc_init(&conv_bias_md, 1, conv_bias_sizes,
        mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_desc_init(&conv_dst_md, 4, conv_dst_sizes,
        mkldnn_f32, mkldnn_any));

    /* create a convolution */
    mkldnn_convolution_desc_t conv_any_desc;
    CHECK(mkldnn_convolution_forward_desc_init(&conv_any_desc, mkldnn_forward,
            mkldnn_convolution_direct, &conv_src_md, &conv_weights_md,
            &conv_bias_md, &conv_dst_md, conv_strides, conv_padding,
            conv_padding, mkldnn_padding_zero));

    mkldnn_primitive_desc_t conv_pd;
    CHECK(mkldnn_primitive_desc_create(&conv_pd, &conv_any_desc,
            engine, NULL));

    mkldnn_primitive_t conv_internal_src_memory, conv_internal_weights_memory,
        conv_internal_dst_memory;

    float *conv_src_buffer =
        (float*)aligned_malloc(product(conv_src_sizes, 4)*sizeof(float), 64);
    float *conv_weights_buffer =
        (float*)aligned_malloc(product(conv_weights_sizes, 4)*sizeof(float), 64);
    float *conv_dst_buffer =
        (float*)aligned_malloc(product(conv_dst_sizes, 4)*sizeof(float), 64);
    memset(conv_src_buffer, 0, product(conv_src_sizes, 4)*sizeof(float));
    memset(conv_weights_buffer, 0, product(conv_weights_sizes, 4)*sizeof(float));
    memset(conv_dst_buffer, 0, product(conv_dst_sizes, 4)*sizeof(float));

    /* create memory for dst data, we don't need reorder it to user data */
    CHECK(mkldnn_primitive_create(&conv_internal_dst_memory,
            mkldnn_primitive_desc_query_pd(conv_pd, mkldnn_query_dst_pd, 0),
            NULL, NULL));
    CHECK(mkldnn_memory_set_data_handle(
            conv_internal_dst_memory, conv_dst_buffer));

    /* create reorder primitives between user data and convolution srcs
     * if required */
    mkldnn_primitive_t conv_reorder_src, conv_reorder_weights;

    const_mkldnn_primitive_desc_t src_pd = mkldnn_primitive_desc_query_pd(
            conv_pd, mkldnn_query_src_pd, 0);
    CHECK(prepare_reorder(&conv_user_src_memory, &src_pd, 1,
            &conv_internal_src_memory, &conv_reorder_src,
            conv_src_buffer));

    const_mkldnn_primitive_desc_t weights_pd = mkldnn_primitive_desc_query_pd(
            conv_pd, mkldnn_query_weights_pd, 0);
    CHECK(prepare_reorder(&conv_user_weights_memory, &weights_pd, 1,
            &conv_internal_weights_memory, &conv_reorder_weights,
            conv_weights_buffer));

    mkldnn_primitive_t conv_src_memory = conv_internal_src_memory ?
        conv_internal_src_memory : conv_user_src_memory;
    mkldnn_primitive_t conv_weights_memory = conv_internal_weights_memory ?
        conv_internal_weights_memory : conv_user_weights_memory;

    mkldnn_primitive_at_t conv_srcs[] = {
        mkldnn_primitive_at(conv_src_memory, 0),
        mkldnn_primitive_at(conv_weights_memory, 0),
        mkldnn_primitive_at(conv_user_bias_memory, 0)
    };

    const_mkldnn_primitive_t conv_dsts[] = { conv_internal_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv;
    CHECK(mkldnn_primitive_create(&conv, conv_pd, conv_srcs, conv_dsts));

    /* AlexNet: relu
     * {BATCH, 96, 55, 55} -> {BATCH, 96, 55, 55}
     */
    float negative_slope = 1.0;

    int *relu_dst_sizes = conv_dst_sizes;
    float *relu_dst_buffer =
        (float*)aligned_malloc(product(relu_dst_sizes, 4)*sizeof(float), 64);
    memset(relu_dst_buffer, 0, product(relu_dst_sizes, 4)*sizeof(float));

    /* create relu memory descriptor on dst memory descriptor
     * from previos primitive */
    const_mkldnn_primitive_desc_t conv_dst_pd = mkldnn_primitive_desc_query_pd(
            conv_pd, mkldnn_query_dst_pd, 0);
    const mkldnn_memory_desc_t *relu_src_md =
        mkldnn_primitive_desc_query_memory_d(conv_dst_pd);

    /* create a relu */
    mkldnn_eltwise_desc_t relu_desc;
    CHECK(mkldnn_eltwise_forward_desc_init(&relu_desc, mkldnn_forward,
                mkldnn_eltwise_relu, relu_src_md, negative_slope, 0));

    mkldnn_primitive_desc_t relu_pd;
    CHECK(mkldnn_primitive_desc_create(&relu_pd, &relu_desc, engine, NULL));

    mkldnn_primitive_t relu_dst_memory;
    const_mkldnn_primitive_desc_t relu_dst_pd = mkldnn_primitive_desc_query_pd(
            relu_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(&relu_dst_memory, relu_dst_pd, NULL, NULL));
    CHECK(mkldnn_memory_set_data_handle(relu_dst_memory, relu_dst_buffer));

    /* finally create a relu primitive */
    mkldnn_primitive_t relu;
    mkldnn_primitive_at_t relu_srcs = { conv_internal_dst_memory, 0 };
    const_mkldnn_primitive_t relu_dsts[] = { relu_dst_memory };

    CHECK(mkldnn_primitive_create(&relu, relu_pd, &relu_srcs, relu_dsts));

    /* AlexNet: lrn
     * {BATCH, 96, 55, 55} -> {BATCH, 96, 55, 55}
     * local size: 5
     * alpha: 0.0001
     * beta: 0.75
     */
    uint32_t local_size = 5;
    float alpha = 0.0001;
    float beta = 0.75;
    float k = 1.0;

    int32_t *lrn_dst_sizes = relu_dst_sizes;

    float *lrn_dst_buffer =
        (float*)aligned_malloc(product(lrn_dst_sizes, 4)*sizeof(float), 64);
    memset(lrn_dst_buffer, 0, product(lrn_dst_sizes, 4)*sizeof(float));

    /* create lrn memory descriptor on dst memory descriptor
     *  from previos primitive */
    const mkldnn_memory_desc_t *lrn_src_md =
        mkldnn_primitive_desc_query_memory_d(relu_dst_pd);

    /* create a lrn */
    mkldnn_lrn_desc_t lrn_desc;
    CHECK(mkldnn_lrn_forward_desc_init(&lrn_desc, mkldnn_forward,
            mkldnn_lrn_across_channels, lrn_src_md, local_size,
            alpha, beta, k));

    mkldnn_primitive_desc_t lrn_pd;
    CHECK(mkldnn_primitive_desc_create(&lrn_pd, &lrn_desc, engine, NULL));

    mkldnn_primitive_t lrn_dst_memory;
    const_mkldnn_primitive_desc_t lrn_dst_pd = mkldnn_primitive_desc_query_pd(
            lrn_pd, mkldnn_query_dst_pd, 0);
    CHECK(mkldnn_primitive_create(&lrn_dst_memory, lrn_dst_pd, NULL, NULL));
    CHECK(mkldnn_memory_set_data_handle(lrn_dst_memory, lrn_dst_buffer));

    mkldnn_primitive_t lrn_scratch_memory;
    const_mkldnn_primitive_desc_t lrn_scratch_pd =
        mkldnn_primitive_desc_query_pd(lrn_pd, mkldnn_query_workspace_pd, 0);
    CHECK(mkldnn_primitive_create(&lrn_scratch_memory,
            lrn_scratch_pd, NULL, NULL));
    size_t lrn_scratch_size =
        mkldnn_memory_primitive_desc_get_size(lrn_scratch_pd);
    float *lrn_scratch_buffer = (float*)aligned_malloc(lrn_scratch_size, 64);
    memset(lrn_scratch_buffer, 0, lrn_scratch_size);
    CHECK(mkldnn_memory_set_data_handle(lrn_scratch_memory,
            lrn_scratch_buffer));

    mkldnn_primitive_at_t lrn_srcs = { relu_dst_memory, 0 };

    const_mkldnn_primitive_t lrn_dsts[] = { lrn_dst_memory,
            lrn_scratch_memory };

    /* finally create a lrn primitive */
    mkldnn_primitive_t lrn;
    CHECK(mkldnn_primitive_create(&lrn, lrn_pd, &lrn_srcs, lrn_dsts));

    /* AlexNet: pool
     * {BATCH, 96, 55, 55} -> {BATCH, 96, 27, 27}
     * kernel: {3, 3}
     * strides: {2, 2}
     */
    int32_t pool_dst_sizes[4] = {BATCH, 96, 27, 27};
    int32_t pool_kernel[2] = {3, 3};
    int32_t pool_strides[2] = {2, 2};
    int32_t pool_padding[2] = {0, 0};

    float *pool_dst_buffer =
        (float*)aligned_malloc(product(pool_dst_sizes, 4)*sizeof(float), 64);
    memset(pool_dst_buffer, 0, product(pool_dst_sizes, 4)*sizeof(float));

    /* create pooling memory descriptor on dst descriptor
     *  from previos primitive */
    const mkldnn_memory_desc_t *pool_src_md =
        mkldnn_primitive_desc_query_memory_d(lrn_dst_pd);

    /* create descriptors for dst pooling data */
    mkldnn_memory_desc_t pool_dst_md;
    CHECK(mkldnn_memory_desc_init(&pool_dst_md, 4, pool_dst_sizes, mkldnn_f32,
            mkldnn_any));

    /* create memory for user data */
    mkldnn_primitive_t pool_user_dst_memory;
    init_data_memory(4, pool_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
        net_dst, &pool_user_dst_memory);

    /* create a pooling */
    mkldnn_pooling_desc_t pool_desc;
    CHECK(mkldnn_pooling_forward_desc_init(&pool_desc, mkldnn_forward,
            mkldnn_pooling_max, pool_src_md, &pool_dst_md, pool_strides,
            pool_kernel, pool_padding, pool_padding, mkldnn_padding_zero));

    mkldnn_primitive_desc_t pool_pd;
    CHECK(mkldnn_primitive_desc_create(&pool_pd, &pool_desc, engine, NULL));

    /* create memory for workspace */
    mkldnn_primitive_t pool_indices_memory;
    const_mkldnn_primitive_desc_t pool_indices_pd =
        mkldnn_primitive_desc_query_pd(pool_pd, mkldnn_query_workspace_pd, 0);
    CHECK(mkldnn_primitive_create(&pool_indices_memory,
            pool_indices_pd, NULL, NULL));
    size_t pool_indices_size =
        mkldnn_memory_primitive_desc_get_size(pool_indices_pd);
    float *pool_indices_buffer = (float*)aligned_malloc(pool_indices_size, 64);
    memset(pool_indices_buffer, 0, pool_indices_size);
    CHECK(mkldnn_memory_set_data_handle(pool_indices_memory,
            pool_indices_buffer));

    mkldnn_primitive_t pool_dst_memory;

    /* create reorder primitives between user data and pooling dsts
     * if required */
    mkldnn_primitive_t pool_reorder_dst, pool_internal_dst_memory;
    const_mkldnn_primitive_desc_t pool_dst_pd =
        mkldnn_primitive_desc_query_pd(pool_pd, mkldnn_query_dst_pd, 0);
    CHECK(prepare_reorder(&pool_user_dst_memory, &pool_dst_pd, 0,
            &pool_internal_dst_memory, &pool_reorder_dst, pool_dst_buffer));

    mkldnn_primitive_at_t pool_srcs = { lrn_dst_memory, 0 };

    pool_dst_memory = pool_internal_dst_memory ? pool_internal_dst_memory
        : pool_user_dst_memory;

    const_mkldnn_primitive_t pool_dsts[] = { pool_dst_memory,
            pool_indices_memory };

    /* finally create a pooling primitive */
    mkldnn_primitive_t pool;
    CHECK(mkldnn_primitive_create(&pool, pool_pd, &pool_srcs, pool_dsts));

    /* build a simple net */
    uint32_t n = 0;
    mkldnn_primitive_t net[10];

    if (conv_reorder_src) net[n++] = conv_reorder_src;
    if (conv_reorder_weights) net[n++] = conv_reorder_weights;
    net[n++] = conv;
    net[n++] = relu;
    net[n++] = lrn;
    net[n++] = pool;
    if (pool_reorder_dst) net[n++] = pool_reorder_dst;

    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream, mkldnn_eager));
    CHECK(mkldnn_stream_submit(stream, n, net, NULL));
    CHECK(mkldnn_stream_wait(stream, n, NULL));

    /* clean-up */
    CHECK(mkldnn_primitive_desc_destroy(conv_pd));
    CHECK(mkldnn_primitive_desc_destroy(relu_pd));
    CHECK(mkldnn_primitive_desc_destroy(lrn_pd));
    CHECK(mkldnn_primitive_desc_destroy(pool_pd));

    mkldnn_stream_destroy(stream);

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

    mkldnn_primitive_destroy(lrn_scratch_memory);
    mkldnn_primitive_destroy(lrn_dst_memory);
    mkldnn_primitive_destroy(lrn);

    _free(lrn_scratch_buffer);
    _free(lrn_dst_buffer);

    mkldnn_primitive_destroy(pool_user_dst_memory);
    mkldnn_primitive_destroy(pool_internal_dst_memory);
    mkldnn_primitive_destroy(pool_indices_memory);
    mkldnn_primitive_destroy(pool_reorder_dst);
    mkldnn_primitive_destroy(pool);

    _free(pool_dst_buffer);
    _free(pool_indices_buffer);

    mkldnn_engine_destroy(engine);

    return mkldnn_success;
}

int main(int argc, char **argv) {
    mkldnn_status_t result = simple_net();
    printf("%s\n", (result == mkldnn_success) ? "passed" : "failed");
    return result;
}
