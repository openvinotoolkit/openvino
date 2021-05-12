// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <c_api/ie_c_api.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Struct to store classification results
 */
struct classify_res {
    size_t class_id;
    float probability;
};

/**
 * @brief Sort result of image classification by probability
 * @param struct with classification results to sort
 * @param size of the struct
 * @return none
 */
void classify_res_sort(struct classify_res* res, size_t n) {
    size_t i, j;
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            if (res[i].probability < res[j].probability) {
                struct classify_res temp = res[i];
                res[i] = res[j];
                res[j] = temp;
            } else if (res[i].probability == res[j].probability && res[i].class_id > res[j].class_id) {
                struct classify_res temp = res[i];
                res[i] = res[j];
                res[j] = temp;
            }
        }
    }
}

/**
 * @brief Convert output blob to classify struct for processing results
 * @param blob of output data
 * @param size of the blob
 * @return struct classify_res
 */
struct classify_res* output_blob_to_classify_res(ie_blob_t* blob, size_t* n) {
    dimensions_t output_dim;
    IEStatusCode status = ie_blob_get_dims(blob, &output_dim);
    if (status != OK)
        return NULL;

    *n = output_dim.dims[1];

    struct classify_res* cls = (struct classify_res*)malloc(sizeof(struct classify_res) * (*n));
    if (!cls) {
        return NULL;
    }

    ie_blob_buffer_t blob_cbuffer;
    status = ie_blob_get_cbuffer(blob, &blob_cbuffer);
    if (status != OK) {
        free(cls);
        return NULL;
    }
    float* blob_data = (float*)(blob_cbuffer.cbuffer);

    size_t i;
    for (i = 0; i < *n; ++i) {
        cls[i].class_id = i;
        cls[i].probability = blob_data[i];
    }

    return cls;
}

/**
 * @brief Print results of classification
 * @param struct of the classification results
 * @param size of the struct of classification results
 * @param string image path
 * @return none
 */
void print_classify_res(struct classify_res* cls, size_t n, const char* img_path) {
    printf("\nImage %s\n", img_path);
    printf("\nclassid probability\n");
    printf("------- -----------\n");
    size_t i;
    for (i = 0; i < n; ++i) {
        printf("%zu       %f\n", cls[i].class_id, cls[i].probability);
    }
    printf("\nThis sample is an API example,"
           " for any performance measurements please use the dedicated benchmark_"
           "app tool\n");
}

/**
 * @brief Read image data
 * @param string image path
 * @param pointer to store image data
 * @param size bytes of image
 * @return total number of elements successfully read, in case of error it
 * doesn't equal to size param
 */
size_t read_image_from_file(const char* img_path, unsigned char* img_data, size_t size) {
    FILE* fp = fopen(img_path, "rb+");
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

/**
 * @brief Check image has supported width and height
 * @param string image size in WIDTHxHEIGHT format
 * @param pointer to image width
 * @param pointer to image height
 * @return bool status True(success) or False(fail)
 */
bool is_supported_image_size(const char* size_str, size_t* width, size_t* height) {
    const char* _size = size_str;
    size_t _width = 0, _height = 0;
    while (_size && *_size != 'x' && *_size != '\0') {
        if ((*_size <= '9') && (*_size >= '0')) {
            _width = (_width * 10) + (*_size - '0');
            _size++;
        } else {
            goto err;
        }
    }

    if (_size)
        _size++;

    while (_size && *_size != '\0') {
        if ((*_size <= '9') && (*_size >= '0')) {
            _height = (_height * 10) + (*_size - '0');
            _size++;
        } else {
            goto err;
        }
    }

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
        goto err;
    }
err:
    printf("Incorrect format of image size parameter, expected WIDTHxHEIGHT, "
           "actual: %s\n",
           size_str);
    return false;
}

int main(int argc, char** argv) {
    // ------------------------------ Parsing and validation of input args
    // ---------------------------------
    if (argc != 5) {
        printf("Usage : ./hello_classification <path_to_model> <path_to_image> "
               "<image_size> <device_name>\n");
        return EXIT_FAILURE;
    }

    size_t input_width = 0, input_height = 0, img_size = 0;
    if (!is_supported_image_size(argv[3], &input_width, &input_height))
        return EXIT_FAILURE;

    const char* input_model = argv[1];
    const char* input_image_path = argv[2];
    const char* device_name = argv[4];
    unsigned char* img_data = NULL;
    ie_core_t* core = NULL;
    ie_network_t* network = NULL;
    ie_executable_network_t* exe_network = NULL;
    ie_infer_request_t* infer_request = NULL;
    char *input_name = NULL, *output_name = NULL;
    ie_blob_t *y_blob = NULL, *uv_blob = NULL, *nv12_blob = NULL, *output_blob = NULL;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 1. Initialize inference engine core
    // -------------------------------------
    IEStatusCode status = ie_core_create("", &core);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin
    // files) or ONNX (.onnx file) format
    status = ie_core_read_network(core, input_model, NULL, &network);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 3. Configure input & output
    // ---------------------------------------------
    // --------------------------- Prepare input blobs
    // -----------------------------------------------------
    status = ie_network_get_input_name(network, 0, &input_name);
    if (status != OK)
        goto err;

    /* Mark input as resizable by setting of a resize algorithm.
     * In this case we will be able to set an input blob of any shape to an infer
     * request. Resize and layout conversions are executed automatically during
     * inference */
    status |= ie_network_set_input_resize_algorithm(network, input_name, RESIZE_BILINEAR);
    status |= ie_network_set_input_layout(network, input_name, NCHW);
    status |= ie_network_set_input_precision(network, input_name, U8);
    // set input color format to NV12 to enable automatic input color format
    // pre-processing
    status |= ie_network_set_color_format(network, input_name, NV12);

    if (status != OK)
        goto err;

    // --------------------------- Prepare output blobs
    // ----------------------------------------------------
    status |= ie_network_get_output_name(network, 0, &output_name);
    status |= ie_network_set_output_precision(network, output_name, FP32);
    if (status != OK)
        goto err;

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 4. Loading model to the device
    // ------------------------------------------
    ie_config_t config = {NULL, NULL, NULL};
    status = ie_core_load_network(core, network, device_name, &config, &exe_network);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 5. Create infer request
    // -------------------------------------------------
    status = ie_exec_network_create_infer_request(exe_network, &infer_request);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 6. Prepare input
    // -------------------------------------------------------- read image with
    // size converted to NV12 data size: height(NV12) = 3 / 2 * logical height
    img_size = input_width * (input_height * 3 / 2);
    img_data = (unsigned char*)calloc(img_size, sizeof(unsigned char));
    if (NULL == img_data)
        goto err;
    if (img_size != read_image_from_file(input_image_path, img_data, img_size))
        goto err;

    // --------------------------- Create a blob to hold the NV12 input data
    // ------------------------------- Create tensor descriptors for Y and UV
    // blobs
    dimensions_t y_dimens = {4, {1, 1, input_height, input_width}};
    dimensions_t uv_dimens = {4, {1, 2, input_height / 2, input_width / 2}};
    tensor_desc_t y_tensor = {NHWC, y_dimens, U8};
    tensor_desc_t uv_tensor = {NHWC, uv_dimens, U8};
    size_t y_plane_size = input_height * input_width;
    size_t uv_plane_size = input_width * (input_height / 2);

    // Create blob for Y plane from raw data
    status |= ie_blob_make_memory_from_preallocated(&y_tensor, img_data, y_plane_size, &y_blob);
    // Create blob for UV plane from raw data
    status |= ie_blob_make_memory_from_preallocated(&uv_tensor, img_data + y_plane_size, uv_plane_size, &uv_blob);
    // Create NV12Blob from Y and UV blobs
    status |= ie_blob_make_memory_nv12(y_blob, uv_blob, &nv12_blob);
    if (status != OK)
        goto err;

    status = ie_infer_request_set_blob(infer_request, input_name, nv12_blob);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 7. Do inference
    // --------------------------------------------------------
    /* Running the request synchronously */
    status = ie_infer_request_infer(infer_request);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 8. Process output
    // ------------------------------------------------------
    status = ie_infer_request_get_blob(infer_request, output_name, &output_blob);
    if (status != OK)
        goto err;

    size_t class_num;
    struct classify_res* cls = output_blob_to_classify_res(output_blob, &class_num);

    classify_res_sort(cls, class_num);

    // Print classification results
    size_t top = 10;
    if (top > class_num) {
        top = class_num;
    }
    printf("\nTop %zu results:\n", top);
    print_classify_res(cls, top, input_image_path);
    // -----------------------------------------------------------------------------------------------------

    free(cls);
    ie_blob_free(&output_blob);
    ie_blob_free(&nv12_blob);
    ie_blob_free(&uv_blob);
    ie_blob_free(&y_blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_name_free(&input_name);
    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);
    free(img_data);

    return EXIT_SUCCESS;
err:
    if (core)
        ie_core_free(&core);
    if (network)
        ie_network_free(&network);
    if (input_name)
        ie_network_name_free(&input_name);
    if (output_name)
        ie_network_name_free(&output_name);
    if (exe_network)
        ie_exec_network_free(&exe_network);
    if (infer_request)
        ie_infer_request_free(&infer_request);
    if (nv12_blob)
        ie_blob_free(&nv12_blob);
    if (uv_blob)
        ie_blob_free(&uv_blob);
    if (y_blob)
        ie_blob_free(&y_blob);
    if (output_blob)
        ie_blob_free(&output_blob);
    if (img_data)
        free(img_data);
    return EXIT_FAILURE;
}
