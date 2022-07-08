// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c_api/ov_c_api.h"

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

/**
 * @brief Convert output tensor to infer result struct for processing results
 * @param tensor of output tensor
 * @param result_size of the infer result
 * @return struct infer_result
 */
struct infer_result* tensor_to_infer_result(ov_tensor_t* tensor, size_t* result_size) {
    ov_status_e status = ov_tensor_get_size(tensor, result_size);
    if (status != OK)
        return NULL;

    struct infer_result* results = (struct infer_result*)malloc(sizeof(struct infer_result) * (*result_size));
    if (!results)
        return NULL;

    void* data = NULL;
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
    ov_free(friendly_name);
}

/**
 * @brief Check image has supported width and height
 * @param string image size in WIDTHxHEIGHT format
 * @param pointer to image width
 * @param pointer to image height
 * @return bool status True(success) or False(fail)
 */

bool is_supported_image_size(const char* size_str, size_t* width, size_t* height) {
    char* p_end = NULL;
    size_t _width = 0, _height = 0;
    _width = strtoul(size_str, &p_end, 10);
    _height = strtoul(p_end + 1, NULL, 10);
    if (_width > 0 && _height > 0) {
        if (_width % 2 == 0 && _height % 2 == 0) {
            *width = _width;
            *height = _height;
            return true;
        } else {
            printf("Unsupported image size, width and height must be even numbers \n");
            return false;
        }
    } else {
        printf("Incorrect format of image size parameter, expected WIDTHxHEIGHT, "
               "actual: %s\n",
               size_str);
        return false;
    }
}

size_t read_image_from_file(const char* img_path, unsigned char* img_data, size_t size) {
    FILE* fp = fopen(img_path, "rb");
    size_t read_size = 0;

    if (fp) {
        fseek(fp, 0, SEEK_END);
        if (ftell(fp) >= size) {
            fseek(fp, 0, SEEK_SET);
            read_size = fread(img_data, 1, size, fp);
        }
        fclose(fp);
    }

    return read_size;
}

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

int main(int argc, char** argv) {
    // -------- Check input parameters --------
    if (argc != 5) {
        printf("Usage : ./hello_classification_ov_c <path_to_model> <path_to_image> "
               "<device_name>\n");
        return EXIT_FAILURE;
    }

    size_t input_width = 0, input_height = 0, img_size = 0;
    if (!is_supported_image_size(argv[3], &input_width, &input_height)) {
        fprintf(stderr, "ERROR is_supported_image_size, line %d\n", __LINE__);
        return EXIT_FAILURE;
    }
    unsigned char* img_data = NULL;
    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_tensor_t* tensor = NULL;
    ov_preprocess_t* preprocess = NULL;
    ov_preprocess_input_info_t* input_info = NULL;
    ov_model_t* new_model = NULL;
    ov_preprocess_input_tensor_info_t* input_tensor_info = NULL;
    ov_preprocess_input_process_steps_t* input_process = NULL;
    ov_preprocess_input_model_info_t* p_input_model = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_infer_request_t* infer_request = NULL;
    ov_tensor_t* output_tensor = NULL;
    struct infer_result* results = NULL;
    char* input_tensor_name;
    char* output_tensor_name;
    ov_output_node_list_t input_nodes;
    ov_output_node_list_t output_nodes;

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
    const char* device_name = argv[4];

    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    CHECK_STATUS(ov_core_create("", &core));

    // -------- Step 2. Read a model --------
    printf("[INFO] Loading model files: %s\n", input_model);
    CHECK_STATUS(ov_core_read_model(core, input_model, NULL, &model));
    print_model_input_output_info(model);

    CHECK_STATUS(ov_model_get_outputs(model, &output_nodes));
    if (output_nodes.num != 1) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 output only %d\n", __LINE__);
        goto err;
    }

    CHECK_STATUS(ov_model_get_inputs(model, &input_nodes));
    if (input_nodes.num != 1) {
        fprintf(stderr, "[ERROR] Sample supports models with 1 input only %d\n", __LINE__);
        goto err;
    }

    CHECK_STATUS(ov_node_get_tensor_name(&input_nodes, 0, &input_tensor_name));
    CHECK_STATUS(ov_node_get_tensor_name(&output_nodes, 0, &output_tensor_name));

    // -------- Step 3. Configure preprocessing  --------
    CHECK_STATUS(ov_preprocess_create(model, &preprocess));

    // 1) Select input with 'input_tensor_name' tensor name
    CHECK_STATUS(ov_preprocess_get_input_info_by_name(preprocess, input_tensor_name, &input_info));

    // 2) Set input type
    // - as 'u8' precision
    // - set color format to NV12 (single plane)
    // - static spatial dimensions for resize preprocessing operation
    CHECK_STATUS(ov_preprocess_input_get_tensor_info(input_info, &input_tensor_info));
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_element_type(input_tensor_info, U8));
    CHECK_STATUS(ov_preprocess_input_tensor_info_set_color_format(input_tensor_info, NV12_SINGLE_PLANE));
    CHECK_STATUS(
        ov_preprocess_input_tensor_info_set_spatial_static_shape(input_tensor_info, input_height, input_width));

    // 3) Pre-processing steps:
    //    a) Convert to 'float'. This is to have color conversion more accurate
    //    b) Convert to BGR: Assumes that model accepts images in BGR format. For RGB, change it manually
    //    c) Resize image from tensor's dimensions to model ones
    CHECK_STATUS(ov_preprocess_input_get_preprocess_steps(input_info, &input_process));
    CHECK_STATUS(ov_preprocess_input_convert_element_type(input_process, F32));
    CHECK_STATUS(ov_preprocess_input_convert_color(input_process, BGR));
    CHECK_STATUS(ov_preprocess_input_resize(input_process, RESIZE_LINEAR));

    // 4) Set model data layout (Assuming model accepts images in NCHW layout)
    CHECK_STATUS(ov_preprocess_input_get_model_info(input_info, &p_input_model));
    ov_layout_t model_layout = {'N', 'C', 'H', 'W'};
    CHECK_STATUS(ov_preprocess_input_model_set_layout(p_input_model, model_layout));

    // 5) Apply preprocessing to an input with 'input_tensor_name' name of loaded model
    CHECK_STATUS(ov_preprocess_build(preprocess, &new_model));

    // -------- Step 4. Loading a model to the device --------
    ov_property_t property;
    CHECK_STATUS(ov_core_compile_model(core, new_model, device_name, &compiled_model, &property));

    // -------- Step 5. Create an infer request --------
    CHECK_STATUS(ov_compiled_model_create_infer_request(compiled_model, &infer_request));

    // -------- Step 6. Prepare input data  --------
    img_size = input_width * (input_height * 3 / 2);
    if (!img_size) {
        fprintf(stderr, "[ERROR] Invalid Image size, line %d\n", __LINE__);
        goto err;
    }
    img_data = (unsigned char*)calloc(img_size, sizeof(unsigned char));
    if (NULL == img_data) {
        fprintf(stderr, "[ERROR] calloc returned NULL, line %d\n", __LINE__);
        goto err;
    }
    if (img_size != read_image_from_file(input_image_path, img_data, img_size)) {
        fprintf(stderr, "[ERROR] Image dimensions not match with NV12 file size, line %d\n", __LINE__);
        goto err;
    }
    ov_element_type_e input_type = U8;
    size_t batch = 1;
    ov_shape_t input_shape = {4, {batch, input_height * 3 / 2, input_width, 1}};
    CHECK_STATUS(ov_tensor_create_from_host_ptr(input_type, input_shape, img_data, &tensor));

    // -------- Step 6. Set input tensor  --------
    // Set the input tensor by tensor name to the InferRequest
    CHECK_STATUS(ov_infer_request_set_tensor(infer_request, input_tensor_name, tensor));

    // -------- Step 7. Do inference --------
    // Running the request synchronously
    CHECK_STATUS(ov_infer_request_infer(infer_request));

    // -------- Step 8. Process output --------
    CHECK_STATUS(ov_infer_request_get_out_tensor(infer_request, 0, &output_tensor));
    // Print classification results
    size_t results_num;
    results = tensor_to_infer_result(output_tensor, &results_num);
    if (!results) {
        goto err;
    }
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
    free(img_data);
    ov_output_node_list_free(&output_nodes);
    ov_output_node_list_free(&input_nodes);
    if (output_tensor)
        ov_tensor_free(output_tensor);
    if (infer_request)
        ov_infer_request_free(infer_request);
    if (compiled_model)
        ov_compiled_model_free(compiled_model);
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
