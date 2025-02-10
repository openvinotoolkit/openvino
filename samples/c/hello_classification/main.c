// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <opencv_c_wrapper.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer_result_util.h"
#include "openvino/c/openvino.h"

void print_model_input_output_info(ov_model_t* model) {
    char* friendly_name = NULL;
    ov_model_get_friendly_name(model, &friendly_name);
    printf("[INFO] model name: %s \n", friendly_name);
    ov_free(friendly_name);
}

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

int main(int argc, char** argv) {
    // -------- Check input parameters --------
    if (argc != 4) {
        printf("Usage : ./hello_classification_c <path_to_model> <path_to_image> "
               "<device_name>\n");
        return EXIT_FAILURE;
    }

    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_tensor_t* tensor = NULL;
    ov_preprocess_prepostprocessor_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_model_t* new_model = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_preprocess_steps_t* input_process = NULL;
    ov_preprocess_input_model_info_t* p_input_model = NULL;
    ov_preprocess_output_info_t* output_info = NULL;
    ov_preprocess_output_tensor_info_t* output_tensor_info = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_infer_request_t* infer_request = NULL;
    ov_tensor_t* output_tensor = NULL;
    struct infer_result* results = NULL;
    ov_layout_t* input_layout = NULL;
    ov_layout_t* model_layout = NULL;
    ov_shape_t input_shape;
    ov_output_const_port_t* output_port = NULL;
    ov_output_const_port_t* input_port = NULL;

    // -------- Get OpenVINO runtime version --------
    ov_version_t version;
    CHECK_STATUS(ov_get_openvino_version(&version));
    printf("---- OpenVINO INFO----\n");
    printf("Description : %s \n", version.description);
    printf("Build number: %s \n", version.buildNumber);
    ov_version_free(&version);

    // -------- Parsing and validation of input arguments --------
    const char* input_model = argv[1];
    const char* input_image_path = argv[2];
    const char* device_name = argv[3];

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    CHECK_STATUS(ov_core_create(&core));

    // -------- Step 2. Read a model --------
    printf("[INFO] Loading model files: %s\n", input_model);
    CHECK_STATUS(ov_core_read_model(core, input_model, NULL, &model));
    print_model_input_output_info(model);

    CHECK_STATUS(ov_model_const_output(model, &output_port));
    if (!output_port) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 output only %d\n", __LINE__);
        goto err;
    }

    CHECK_STATUS(ov_model_const_input(model, &input_port));
    if (!input_port) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 input only %d\n", __LINE__);
        goto err;
    }

    // -------- Step 3. Set up input
    c_mat_t img;
    image_read(input_image_path, &img);
    ov_element_type_e input_type = U8;
    int64_t dims[4] = {1, (size_t)img.mat_height, (size_t)img.mat_width, 3};
    ov_shape_create(4, dims, &input_shape);
    CHECK_STATUS(ov_tensor_create_from_host_ptr(input_type, input_shape, img.mat_data, &tensor));

    // -------- Step 4. Configure preprocessing --------
    CHECK_STATUS(ov_preprocess_prepostprocessor_create(model, &preprocess));
    CHECK_STATUS(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));

    CHECK_STATUS(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_from(input_tensor_info, tensor));

    const char* input_layout_desc = "NHWC";
    CHECK_STATUS(ov_layout_create(input_layout_desc, &input_layout));
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, input_layout));

    CHECK_STATUS(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
    CHECK_STATUS(ov_preprocess_preprocess_steps_resize(input_process, RESIZE_LINEAR));
    CHECK_STATUS(ov_preprocess_input_info_get_model_info(input_info, &p_input_model));

    const char* model_layout_desc = "NCHW";
    CHECK_STATUS(ov_layout_create(model_layout_desc, &model_layout));
    CHECK_STATUS(ov_preprocess_input_model_info_set_layout(p_input_model, model_layout));

    CHECK_STATUS(ov_preprocess_prepostprocessor_get_output_info_by_index(preprocess, 0, &output_info));
    CHECK_STATUS(ov_preprocess_output_info_get_tensor_info(output_info, &output_tensor_info));
    CHECK_STATUS(ov_preprocess_output_set_element_type(output_tensor_info, F32));

    CHECK_STATUS(ov_preprocess_prepostprocessor_build(preprocess, &new_model));

    // -------- Step 5. Loading a model to the device --------
    CHECK_STATUS(ov_core_compile_model(core, new_model, device_name, 0, &compiled_model));

    // -------- Step 6. Create an infer request --------
    CHECK_STATUS(ov_compiled_model_create_infer_request(compiled_model, &infer_request));

    // -------- Step 7. Prepare input --------
    CHECK_STATUS(ov_infer_request_set_input_tensor_by_index(infer_request, 0, tensor));

    // -------- Step 8. Do inference synchronously --------
    CHECK_STATUS(ov_infer_request_infer(infer_request));

    // -------- Step 9. Process output
    CHECK_STATUS(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
    // Print classification results
    size_t results_num;
    results = tensor_to_infer_result(output_tensor, &results_num);
    infer_result_sort(results, results_num);
    size_t top = 10;
    if (top > results_num) {
        top = results_num;
    }
    printf("\nTop %zu results:\n", top);
    print_infer_result(results, top, input_image_path);

    // -------- free allocated resources --------
err:
    free(results);
    image_free(&img);
    ov_shape_free(&input_shape);
    ov_output_const_port_free(output_port);
    ov_output_const_port_free(input_port);
    if (output_tensor)
        ov_tensor_free(output_tensor);
    if (infer_request)
        ov_infer_request_free(infer_request);
    if (compiled_model)
        ov_compiled_model_free(compiled_model);
    if (input_layout)
        ov_layout_free(input_layout);
    if (model_layout)
        ov_layout_free(model_layout);
    if (output_tensor_info)
        ov_preprocess_output_tensor_info_free(output_tensor_info);
    if (output_info)
        ov_preprocess_output_info_free(output_info);
    if (p_input_model)
        ov_preprocess_input_model_info_free(p_input_model);
    if (input_process)
        ov_preprocess_preprocess_steps_free(input_process);
    if (input_tensor_info)
        ov_preprocess_input_tensor_info_free(input_tensor_info);
    if (input_info)
        ov_preprocess_input_info_free(input_info);
    if (preprocess)
        ov_preprocess_prepostprocessor_free(preprocess);
    if (new_model)
        ov_model_free(new_model);
    if (tensor)
        ov_tensor_free(tensor);
    if (model)
        ov_model_free(model);
    if (core)
        ov_core_free(core);
    return EXIT_SUCCESS;
}
