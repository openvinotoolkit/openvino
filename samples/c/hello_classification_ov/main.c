// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "c_api/ov_c_api.h"
#include <opencv_c_wrapper.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
void infer_result_sort(struct infer_result* results, size_t result_size) {
    size_t i, j;
    for (i = 0; i < result_size; ++i) {
        for (j = i + 1; j < result_size; ++j) {
            if (results[i].probability < results[j].probability) {
                struct infer_result temp = results[i];
                results[i] = results[j];
                results[j] = temp;
            } else if (results[i].probability == results[j].probability && results[i].class_id > results[j].class_id) {
                struct infer_result temp = results[i];
                results[i] = results[j];
                results[j] = temp;
            }
        }
    }
}

/**
 * @brief Convert output tensor to infer result struct for processing results
 * @param tensor of output tensor
 * @param result_size of the infer result
 * @return struct infer_result
 */
struct infer_result* tensor_to_infer_result(ov_tensor_t* tensor, size_t* result_size) {
    ov_shape_t output_shape = {-1, -1, -1, -1};
    ov_status_e status = ov_tensor_get_shape(tensor, &output_shape);
    if (status != OK) return NULL;

    *result_size = output_shape[1];

    struct infer_result* results = (struct infer_result*)malloc(sizeof(struct infer_result) * (*result_size));
    if (!results) return NULL;

    void *data = NULL;
    status = ov_tensor_get_data(tensor, &data);
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

    return results;
}

/**
 * @brief Print results of infer
 * @param results of the infer results
 * @param result_size of the struct of classification results
 * @param img_path image path
 * @return none
 */
void print_infer_result(struct infer_result* results, size_t result_size, const char* img_path) {
    printf("\nImage %s\n", img_path);
    printf("\nclassid probability\n");
    printf("------- -----------\n");
    size_t i;
    for (i = 0; i < result_size; ++i) {
        printf("%zu       %f\n", results[i].class_id, results[i].probability);
    }
}

void print_model_input_output_info(ov_model_t* model) {
    char* friendly_name = NULL;
    ov_model_get_friendly_name(model, &friendly_name);
    printf("[INFO] model name: %s \n", friendly_name);
    ov_name_free(friendly_name);
}

#define CHECK_STATUS(return_status) if(return_status !=OK) {fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); goto err;}

int main(int argc, char** argv) {
    // -------- Check input parameters --------
    if (argc != 4) {
        printf("Usage : ./hello_classification_ov_c <path_to_model> <path_to_image> "
               "<device_name>\n");
        return EXIT_FAILURE;
    }

    // -------- Get OpenVINO runtime version --------
    ov_version_t version;
    CHECK_STATUS(ov_get_version(&version));
    printf("---- OpenVINO INFO----\n");
    printf("description : %s \n", version.description);
    printf("build number: %s \n", version.buildNumber);
    ov_version_free(&version);

    // -------- Parsing and validation of input arguments --------
    const char* input_model = argv[1];
    const char* input_image_path = argv[2];
    const char* device_name = argv[3];

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov_core_t* core = NULL;
    CHECK_STATUS(ov_core_create("", &core));

    // -------- Step 2. Read a model --------
    printf("[INFO] Loading model files: %s\n", input_model);
    ov_model_t* model = NULL;
    CHECK_STATUS(ov_core_read_model(core, input_model, NULL, &model));
    print_model_input_output_info(model);

    ov_output_node_list_t output_nodes;
    CHECK_STATUS(ov_model_get_outputs(model, &output_nodes));
    if (output_nodes.num != 1) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 output only %d\n", __LINE__);
        goto err;
    }
    ov_output_node_list_t input_nodes;
    CHECK_STATUS(ov_model_get_inputs(model, &input_nodes));
    if (input_nodes.num != 1) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 input only %d\n", __LINE__);
        goto err;
    }
    
    // -------- Step 3. Set up input
    c_mat_t img;
    image_read(input_image_path, &img);
    ov_element_type_e input_type = U8;
    ov_shape_t input_shape = {1, (size_t)img.mat_height, (size_t)img.mat_width, 3};
    ov_tensor_t* tensor = NULL;
    CHECK_STATUS(ov_tensor_create_from_host_ptr(input_type, input_shape, img.mat_data, &tensor));

    // -------- Step 4. Configure preprocessing --------
    ov_preprocess_t* preprocess = NULL;
    CHECK_STATUS(ov_preprocess_create(model, &preprocess));

    ov_preprocess_input_info_t* input_info = NULL;
    CHECK_STATUS(ov_preprocess_get_input_info_by_index(preprocess, 0, &input_info));

    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    CHECK_STATUS(ov_preprocess_input_get_tensor_info(input_info, &input_tensor_info));
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_tensor(input_tensor_info, tensor));
    ov_layout_t tensor_layout = {'N', 'H', 'W', 'C'};
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, tensor_layout));

    ov_preprocess_input_process_steps_t* input_process = NULL;
    CHECK_STATUS(ov_preprocess_input_get_preprocess_steps(input_info, &input_process));
    CHECK_STATUS(ov_preprocess_input_resize(input_process, RESIZE_LINEAR));

    ov_preprocess_input_model_info_t* p_input_model = NULL;
    CHECK_STATUS(ov_preprocess_input_get_model_info(input_info, &p_input_model));
    ov_layout_t model_layout = {'N', 'C', 'H', 'W'};
    CHECK_STATUS(ov_preprocess_input_model_set_layout(p_input_model, model_layout));

    ov_preprocess_output_info_t* output_info = NULL;
    CHECK_STATUS(ov_preprocess_get_output_info_by_index(preprocess, 0, &output_info));
    ov_preprocess_output_tensor_info_t* output_tensor_info = NULL;
    CHECK_STATUS(ov_preprocess_output_get_tensor_info(output_info, &output_tensor_info));
    CHECK_STATUS(ov_preprocess_output_set_element_type(output_tensor_info, F32));

    ov_model_t* new_model = NULL;
    CHECK_STATUS(ov_preprocess_build(preprocess, &new_model));

    // -------- Step 5. Loading a model to the device --------
    ov_compiled_model_t* compiled_model = NULL;
    ov_property_t property = {};
    CHECK_STATUS(ov_core_compile_model(core, new_model, device_name, &compiled_model, &property));

    // -------- Step 6. Create an infer request --------
    ov_infer_request_t *infer_request = NULL;
    CHECK_STATUS(ov_compiled_model_create_infer_request(compiled_model, &infer_request));

    // -------- Step 7. Prepare input --------
    CHECK_STATUS(ov_infer_request_set_input_tensor(infer_request, 0, tensor));

    // -------- Step 8. Do inference synchronously --------
    CHECK_STATUS(ov_infer_request_infer(infer_request));

    // -------- Step 9. Process output
    ov_tensor_t* output_tensor = NULL;
    CHECK_STATUS(ov_infer_request_get_out_tensor(infer_request, 0, &output_tensor));
    // Print classification results
    size_t results_num;
    struct infer_result* results = tensor_to_infer_result(output_tensor, &results_num);
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
    ov_output_node_list_free(&output_nodes);
    ov_output_node_list_free(&input_nodes);
    if (output_tensor)
        ov_tensor_free(output_tensor);
    if (infer_request)
        ov_infer_request_free(infer_request);
    if (compiled_model)
        ov_compiled_model_free(compiled_model);
    if (output_tensor_info)
        ov_preprocess_output_tensor_info_free(output_tensor_info);
    if (output_info)
        ov_preprocess_output_info_free(output_info);
    if (p_input_model)
        ov_preprocess_input_model_info_free(p_input_model);
    if (input_process)
        ov_preprocess_input_process_steps_free(input_process);
    if (input_tensor_info)
        ov_preprocess_input_tensor_info_free(input_tensor_info);
    if (input_info)
        ov_preprocess_input_info_free(input_info);
    if (preprocess)
        ov_preprocess_free(preprocess);
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
