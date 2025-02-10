// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/c/openvino.h>
#include <stdlib.h>

/**
 * @brief Struct to store infer results
 */
struct infer_result {
    size_t class_id;
    float probability;
};

/**
 * @brief Sort result by probability
 * @param struct with infer results to sort
 * @param result_size of the struct
 * @return none
 */
int compare(const void* a, const void* b);

void infer_result_sort(struct infer_result* results, size_t result_size);

/**
 * @brief Convert output tensor to infer result struct for processing results
 * @param tensor of output tensor
 * @param result_size of the infer result
 * @return struct infer_result
 */
struct infer_result* tensor_to_infer_result(ov_tensor_t* tensor, size_t* result_size);

/**
 * @brief Print results of infer
 * @param results of the infer results
 * @param result_size of the struct of classification results
 * @param img_path image path
 * @return none
 */
void print_infer_result(struct infer_result* results, size_t result_size, const char* img_path);
