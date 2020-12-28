// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier : Apache-2.0
//

#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <c_api/ie_c_api.h>
#include <opencv_c_wraper.h>

#define BATCH_SIZE_LIMIT 20

const char *info = "[ INFO ] ";
const char *warn = "[ WARNING ] ";

struct classify_res {
    size_t class_id;
    float probability;
};

void classify_res_sort(struct classify_res *res, size_t n, size_t qty) {
    for (size_t q = 0; q < qty; ++q) {
        size_t i, j;
        for (i = 0; i < n; ++i) {
            for (j = i + 1; j < n; ++j) {
                if (res[q*n + i].probability < res[q*n + j].probability) {
                    struct classify_res temp = res[q*n + i];
                    res[q*n + i] = res[q*n + j];
                    res[q*n + j] = temp;
                } else if (res[q*n + i].probability == res[q*n + j].probability && res[q*n + i].class_id > res[q*n + j].class_id) {
                    struct classify_res temp = res[i];
                    res[q*n + i] = res[q*n + j];
                    res[q*n + j] = temp;
                }
            }
        }
    }
}

struct classify_res *output_blob_to_classify_res(ie_blob_t *blob, size_t *n) {
    dimensions_t output_dim;
    IEStatusCode status = ie_blob_get_dims(blob, &output_dim);
    if (status != OK)
        return NULL;

    size_t qty =  output_dim.dims[0];
    *n = output_dim.dims[1];

    struct classify_res *cls = (struct classify_res *)malloc(sizeof(struct classify_res) * (*n) * qty);

    ie_blob_buffer_t blob_cbuffer;
    status = ie_blob_get_cbuffer(blob, &blob_cbuffer);
    if (status != OK) {
        free(cls);
        return NULL;
    }
    float *blob_data = (float*) (blob_cbuffer.cbuffer);

    for (size_t q = 0; q < qty; ++q) {
        for (size_t i = 0; i < *n; ++i) {
            cls[q * (*n) + i].class_id = i;
            cls[q * (*n) + i].probability = blob_data[q * (*n) + i];
        }
    }

    return cls;
}

void print_classify_res(struct classify_res *cls, size_t n, const char **img_path, size_t num, size_t qty) {
    for (size_t q = 0; q < qty; ++q) {
        printf("\nImage %s\n", img_path[q]);
        printf("\nclassid probability\n");
        printf("------- -----------\n");
        for (size_t i = 0; i < n; ++i) {
            printf("%zu       %f\n", cls[q*num + i].class_id, cls[q*num + i].probability);
        }
    }
}

int main(int argc, char **argv) {
    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (argc < 4) {
        printf("Usage : ./hello_classification <path_to_model> <path_to_image_1> ... <path_to_image_n> <device_name>\n");
        return EXIT_FAILURE;
    }
    const char *input_model = argv[1];
    const char *device_name = argv[argc - 1];
    int file_num = argc - 3;
    const char **file_paths = (const char **) calloc(file_num, sizeof(char *));
    printf("%sNumber of images: %d\n", info, file_num);
    for (int i = 0; i < file_num; ++i) {
        file_paths[i] = argv[2+i];
    }
    for (int i = 0; i < file_num; ++i) {
        printf("        File #%d: %s\n", i, file_paths[i]);
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 1. Load inference engine instance ---------------------------------------
    ie_core_t *core = NULL;
    IEStatusCode status = ie_core_create("", &core);
    if (status != OK)
        return EXIT_FAILURE;
    // -----------------------------------------------------------------------------------------------------

    // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
    ie_network_t *network = NULL;
    status = ie_core_read_network(core, input_model, NULL, &network);
    if (status != OK)
        return EXIT_FAILURE;

//    status = ie_network_set_batch_size(network, BATCH_SIZE_LIMIT);
//    if (status != OK)
//        return EXIT_FAILURE;
//    printf("%sNetwork batch size limit is set to %d\n", info, BATCH_SIZE_LIMIT);

    input_shapes_t input_shapes;
    ie_network_get_input_shapes(network, &input_shapes);
    input_shapes.shapes[0].shape.dims[0] = BATCH_SIZE_LIMIT;
    status = ie_network_reshape(network, input_shapes);
    if (status != OK)
        return EXIT_FAILURE;
    printf("%sNetwork batch size limit is set to %d\n", info, BATCH_SIZE_LIMIT);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Configure input blobs ------------------------------------------------
    char *input_name = NULL;
    status = ie_network_get_input_name(network, 0, &input_name);
    if (status != OK)
        return EXIT_FAILURE;
    // setting input layout to avoid color channels transposing when putting input images to input blob
    status |= ie_network_set_input_layout(network, input_name, NHWC);
    status |= ie_network_set_input_precision(network, input_name, U8);
    if (status != OK)
        return EXIT_FAILURE;

    char *name = NULL;
    status |= ie_network_get_input_name(network, 0, &name);
    dimensions_t input_dim;
    status |= ie_network_get_input_dims(network, name, &input_dim);
    if (status != OK)
        return EXIT_FAILURE;
    size_t input_width = input_dim.dims[2], input_height = input_dim.dims[3];

    // --------------------------- 4. Configure output blobs -----------------------------------------------
    char *output_name = NULL;
    status |= ie_network_get_output_name(network, 0, &output_name);
    status |= ie_network_set_output_precision(network, output_name, FP32);
    if (status !=OK)
        return EXIT_FAILURE;
    // -----------------------------------------------------------------------------------------------------

    // ---------------- 5. Loading model to the device with support for dynamic batching -------------------
    ie_executable_network_t *exe_network = NULL;
    ie_config_t config = {"DYN_BATCH_ENABLED", "YES", NULL};
    status = ie_core_load_network(core, network, device_name, &config, &exe_network);
    if (status != OK)
        return EXIT_FAILURE;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Create infer request and set its batch size --------------------------
    ie_infer_request_t *infer_request = NULL;
    status = ie_exec_network_create_infer_request(exe_network, &infer_request);
    if (status != OK)
        return EXIT_FAILURE;
    size_t batch_size = file_num;
    status = ie_infer_request_set_batch(infer_request, batch_size);
    if (status != OK)
        return EXIT_FAILURE;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Prepare input --------------------------------------------------------
    // Collect images data
    c_mat_t *images = (c_mat_t *)calloc(file_num, sizeof(c_mat_t));
    int image_num = 0;
    c_mat_t img = {NULL, 0, 0, 0, 0, 0};
    for (int i = 0; i < file_num; ++i) {
        if (image_read(file_paths[i], &img) == -1) {
            printf("%sImage %s cannot be read!\n", warn, file_paths[i]);
            continue;
        }
        // Store image data
        c_mat_t resized_img = {NULL, 0, 0, 0, 0, 0};
        if ((input_width == img.mat_width) && (input_height == img.mat_height)) {
            resized_img.mat_data_size = img.mat_data_size;
            resized_img.mat_channels = img.mat_channels;
            resized_img.mat_width = img.mat_width;
            resized_img.mat_height = img.mat_height;
            resized_img.mat_type = img.mat_type;
            resized_img.mat_data = calloc(1, resized_img.mat_data_size);
            for (size_t j = 0; j < resized_img.mat_data_size; ++j)
                resized_img.mat_data[j] = img.mat_data[j];
        } else {
            image_resize(&img, &resized_img, (int)input_width, (int)input_height);
            printf("%sImage is resized from (%d, %d) to (%d, %d)\n", \
            warn, img.mat_width, img.mat_height, resized_img.mat_width, resized_img.mat_height);
        }

        if (resized_img.mat_data) {
            images[image_num] = resized_img;
            ++image_num;
        }
    }
    image_free(&img);

    // Creating input blob
    ie_blob_t *imageInput = NULL;
    status = ie_infer_request_get_blob(infer_request, input_name, &imageInput);
    if (status != OK)
        return EXIT_FAILURE;

    dimensions_t input_tensor_dims;
    status = ie_blob_get_dims(imageInput, &input_tensor_dims);
    if (status != OK)
        return EXIT_FAILURE;
    size_t num_channels = input_tensor_dims.dims[1];
    size_t image_size = input_tensor_dims.dims[3] * input_tensor_dims.dims[2];

    // Filling input tensor with images. Channels are in RGB order
    ie_blob_buffer_t blob_buffer;
    status = ie_blob_get_buffer(imageInput, &blob_buffer);
    if (status != OK)
        return EXIT_FAILURE;
    unsigned char *data = (unsigned char *)(blob_buffer.buffer);
    // Iterate over all input images
    int image_id, pid, ch;
    for (image_id = 0; image_id < batch_size; ++image_id) {
        // Iterate over all pixel in image (b,g,r)
        for (pid = 0; pid < image_size; ++pid) {
            // Iterate over all channels of pixel
            for (ch = 0; ch < num_channels; ++ch) {
                // [images stride + channels stride + pixel id ] all in bytes
                data[image_id * image_size * num_channels + pid * num_channels + ch] =
                    images[image_id].mat_data[pid * num_channels + ch];
            }
        }
        image_free(&images[image_id]);
    }
    free(images);
    ie_blob_free(&imageInput);
    // ----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Do inference --------------------------------------------------------
    status = ie_infer_request_infer(infer_request);
    if (status != OK)
        return EXIT_FAILURE;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 9. Process output -------------------------------------------------------
    // Read classification results from output blob
    ie_blob_t *imgBlob = NULL, *output_blob = NULL;
    status = ie_infer_request_get_blob(infer_request, output_name, &output_blob);
    if (status != OK) {
        return EXIT_FAILURE;
    }

    // Parse output blob to struct holding classification results in user-friendly format
    size_t class_num;
    struct classify_res *cls = output_blob_to_classify_res(output_blob, &class_num);
    classify_res_sort(cls, class_num, image_num);

    // Print classification results
    size_t top = 10;
    if (top > class_num) {
        top = class_num;
    }
    printf("\nTop %zu results:\n", top);
    print_classify_res(cls, top, file_paths, class_num, image_num);

    // ---------------------------- 10. Clean up memory and terminate program ------------------------------
    free(cls);
    ie_blob_free(&output_blob);
    ie_blob_free(&imgBlob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_name_free(&input_name);
    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);

    return EXIT_SUCCESS;
}
