// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_result_util.h"

int compare(const void* a, const void* b) {
    const struct infer_result* sa = (const struct infer_result*)a;
    const struct infer_result* sb = (const struct infer_result*)b;
    if (sa->probability < sb->probability) {
        return 1;
    } else if ((sa->probability == sb->probability) && (sa->class_id > sb->class_id)) {
        return 1;
    } else if (sa->probability > sb->probability) {
        return -1;
    }
    return 0;
}
void infer_result_sort(struct infer_result* results, size_t result_size) {
    qsort(results, result_size, sizeof(struct infer_result), compare);
}

struct infer_result* tensor_to_infer_result(ov_tensor_t* tensor, size_t* result_size) {
    ov_shape_t output_shape = {0};
    ov_status_e status = ov_tensor_get_shape(tensor, &output_shape);
    if (status != OK)
        return NULL;

    *result_size = output_shape.dims[1];

    struct infer_result* results = (struct infer_result*)malloc(sizeof(struct infer_result) * (*result_size));
    if (!results)
        return NULL;

    void* data = NULL;
    status = ov_tensor_data(tensor, &data);
    if (status != OK) {
        free(results);
        return NULL;
    }
    float* float_data = (float*)(data);

    size_t i;
    for (i = 0; i < *result_size; ++i) {
        results[i].class_id = i;
        results[i].probability = float_data[i];
    }

    ov_shape_free(&output_shape);
    return results;
}

void print_infer_result(struct infer_result* results, size_t result_size, const char* img_path) {
    printf("\nImage %s\n", img_path);
    printf("\nclassid probability\n");
    printf("------- -----------\n");
    size_t i;
    for (i = 0; i < result_size; ++i) {
        printf("%zu       %f\n", results[i].class_id, results[i].probability);
    }
}
