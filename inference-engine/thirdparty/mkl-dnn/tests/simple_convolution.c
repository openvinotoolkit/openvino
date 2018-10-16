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
    status_t s = f; \
    if (s != success) { \
        printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while(0)

static int product(int dims, int *arr) {
    int res = 1;
    for (int d = 0; d < dims; ++d) res *= arr[d];
    return res;
}

static void init_engine(engine_t *engine) {
    CHECK(engine_create(engine, engine_kind_cpu, 0 /* idx */));
}

static void init_data_desc(memory_primitive_desc_t *user_prim_desc,
        memory_desc_t *prim_md, int ndims_batch, int ndims_channels,
        int ndims_spatial, int *dims, memory_format_t user_fmt,
        const_dnn_engine_t engine)
{
    memory_desc_t user_md;
    tensor_desc_t tensor;
    CHECK(tensor_desc_init(&tensor, ndims_batch, ndims_channels, ndims_spatial,
                dims));

    CHECK(memory_desc_init(&user_md, &tensor, user_fmt));
    CHECK(memory_desc_init(prim_md, &tensor, memory_format_any));

    CHECK(memory_primitive_desc_init(user_prim_desc, &user_md, engine));
}

/* create reorder if required */
status_t prepare_reorder(
        const_dnn_primitive_t user_memory, /** in */
        memory_primitive_desc_t *prim_memory_pd, /** in */
        int dir_is_user_to_prim, /** in: user -> prim or prim -> user */
        dnn_primitive_t *prim_memory, /** out: memory primitive created */
        dnn_primitive_t *reorder /** out: reorder primitive created */
        )
{
    memory_primitive_desc_t user_memory_pd;
    memory_get_primitive_desc(user_memory, &user_memory_pd);

    if (!memory_primitive_desc_equal(&user_memory_pd, prim_memory_pd)) {
        /* memory_create(&p, m, NULL) means allocate memory */
        CHECK(memory_create(prim_memory, prim_memory_pd, NULL));
        reorder_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            /* reorder primitive descriptor doesn't need engine, because it is
             * already appeared in in- and out- memory primitive descriptors */
            CHECK(reorder_primitive_desc_init(&reorder_pd, &user_memory_pd,
                        prim_memory_pd));
            CHECK(reorder_create(reorder, &reorder_pd, user_memory,
                        *prim_memory));
        } else {
            CHECK(reorder_primitive_desc_init(&reorder_pd, prim_memory_pd,
                        &user_memory_pd));
            CHECK(reorder_create(reorder, &reorder_pd, *prim_memory,
                        user_memory));
        }
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return success;
}

int doit() {
    int enough = 8*1024*1024;
    real_t *src = (real_t*)malloc(sizeof(real_t)*enough);
    real_t *weights = (real_t*)malloc(sizeof(real_t)*enough);
    real_t *bias = (real_t*)malloc(sizeof(real_t)*1024);
    real_t *dst = (real_t*)malloc(sizeof(real_t)*enough);

    /* AlexNet: c1
     * {256, 3, 227, 227} (x) {96, 3, 11, 11} -> {256, 96, 55, 55}
     * strides: {4, 4}
     */

    int c1_src_sizes[4] = {256, 3, 227, 227};
    int c1_weights_sizes[4] = {96, 3, 11, 11};
    int c1_bias_sizes[1] = {96};
    int c1_dst_sizes[4] = {256, 96, 55, 55};
    int strides[] = {4, 4};
    int padding[] = {0, 0};

    dnn_engine_t engine;
    init_engine(&engine);

    /* first describe user data and create data descriptors for future
     * convolution w/ no specified format -- we are open to do reorder */
    memory_primitive_desc_t user_c1_src_prim_desc, user_c1_weights_prim_desc,
                            user_c1_bias_prim_desc, user_c1_dst_prim_desc;
    memory_desc_t c1_src_any_md, c1_weights_any_md, c1_bias_any_md,
                  c1_dst_any_md;

    init_data_desc(&user_c1_src_prim_desc, &c1_src_any_md, 1, 1, 2,
            c1_src_sizes, memory_format_nchw, engine);
    init_data_desc(&user_c1_weights_prim_desc, &c1_weights_any_md, 0, 2, 2,
            c1_weights_sizes, memory_format_oihw, engine);
    init_data_desc(&user_c1_bias_prim_desc, &c1_bias_any_md, 1, 0, 0,
            c1_bias_sizes, memory_format_n, engine);
    init_data_desc(&user_c1_dst_prim_desc, &c1_dst_any_md, 1, 1, 2,
            c1_dst_sizes, memory_format_nchw, engine);

    /* create memory for user data */
    dnn_primitive_t user_c1_src_memory, user_c1_weights_memory,
                user_c1_bias_memory, user_c1_dst_memory;
    CHECK(memory_create(&user_c1_src_memory, &user_c1_src_prim_desc,
                src));
    CHECK(memory_create(&user_c1_weights_memory, &user_c1_weights_prim_desc,
                weights));
    CHECK(memory_create(&user_c1_bias_memory, &user_c1_bias_prim_desc, bias));
    CHECK(memory_create(&user_c1_dst_memory, &user_c1_dst_prim_desc,
                dst));

    /** imagine we want convolution to take bias in exactly user-giver format */
    memory_primitive_desc_t user_c1_bias_primitive_desc;
    CHECK(memory_get_primitive_desc(user_c1_bias_memory,
            &user_c1_bias_primitive_desc));

    /* create fwd convolution descriptor with arbitrary data formats, except
     * for bias -- it is expected that bias won't be reorder (only here)
     * we want the fastest convolution to be used */
    convolution_desc_t c1_any_desc;
    CHECK(convolution_forward_desc_init(&c1_any_desc, forward,
                convolution_direct, &c1_src_any_md, &c1_weights_any_md,
                &user_c1_bias_primitive_desc.memory_descriptor,
                &c1_dst_any_md, strides, padding, padding_kind_zero));

    /* XXX: rephrase it, i don't know english :(
     * create fwd convolution primitive descriptor. for given fwd convolution
     * descriptor and engine it produces primitive descriptor with particular
     * src/dst data formats. if fwd convolution descriptor contains 'any'
     * src/dst formats the best convolution primitive would be picked up */
    convolution_primitive_desc_t c1_pd;
    CHECK(convolution_forward_primitive_desc_init(&c1_pd, &c1_any_desc,
                engine));

    /* create reorder primitives between user data and convolution srcs and
     * dsts if required */
    dnn_primitive_t reorder_c1_src, reorder_c1_weights, reorder_c1_dst;
    dnn_primitive_t c1_src_memory, c1_weights_memory, c1_dst_memory;

    CHECK(prepare_reorder(user_c1_src_memory, &c1_pd.src_primitive_desc,
                1, &c1_src_memory, &reorder_c1_src));
    CHECK(prepare_reorder(user_c1_weights_memory, &c1_pd.weights_primitive_desc,
                1, &c1_weights_memory, &reorder_c1_weights));
    CHECK(prepare_reorder(user_c1_dst_memory, &c1_pd.dst_primitive_desc,
                0, &c1_dst_memory, &reorder_c1_dst));

    /* finally create a convolution primitive */
    dnn_primitive_t c1;
    CHECK(convolution_forward_create(&c1, &c1_pd,
                (c1_src_memory ? c1_src_memory : user_c1_src_memory),
                (c1_weights_memory ? c1_weights_memory : user_c1_weights_memory),
                user_c1_bias_memory,
                (c1_dst_memory ? c1_dst_memory : user_c1_dst_memory)));

    /* let us build a net */
    dnn_stream_t stream;
    CHECK(stream_create(&stream));
    dnn_primitive_t net[10], error_primitive;
    size_t n = 0;
    {
        if (reorder_c1_src) net[n++] = reorder_c1_src;
        if (reorder_c1_weights) net[n++] = reorder_c1_weights;
        net[n++] = c1;
        if (reorder_c1_dst) net[n++] = reorder_c1_dst;
    }

    /* actual computations */
    CHECK(stream_submit(stream, n, net, NULL));
    CHECK(stream_wait(stream, 1, NULL));

    /* clean up starts here */
    CHECK(stream_destroy(stream));

    /* primitive_destroy(NULL) is safe */
    primitive_destroy(reorder_c1_src);
    primitive_destroy(reorder_c1_weights);
    primitive_destroy(c1);
    primitive_destroy(reorder_c1_dst);

    /* primitive_destroy(memory created with NULL) would deallocate internal
     * memory */
    primitive_destroy(c1_src_memory);
    primitive_destroy(c1_weights_memory);
    primitive_destroy(c1_dst_memory);

    /* primitive_destroy(user_memory) should not deallocate external memory */
    primitive_destroy(user_c1_src_memory);
    primitive_destroy(user_c1_weights_memory);
    primitive_destroy(user_c1_bias_memory);
    primitive_destroy(user_c1_dst_memory);

    engine_destroy(engine);

    free(src);
    free(weights);
    free(bias);
    free(dst);

    return 0;
}

int main(int argc, char **argv) {
    int rc = doit();
    printf("%s\n", rc ? "failed" : "passed");
    return rc;
}
