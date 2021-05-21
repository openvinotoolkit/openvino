// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <c_api/ie_c_api.h>
#include <opencv_c_wrapper.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "object_detection_sample_ssd.h"

#ifdef _WIN32
    #include "c_w_dirent.h"
#else
    #include <dirent.h>
#endif

#define MAX_IMAGES 20

static const char* img_msg = NULL;
static const char* input_model = NULL;
static const char* device_name = "CPU";
static const char* custom_plugin_cfg_msg = NULL;
static const char* custom_ex_library_msg = NULL;
static const char* config_msg = NULL;
static int file_num = 0;
static char** file_paths = NULL;

const char* info = "[ INFO ] ";
const char* warn = "[ WARNING ] ";

/**
 * @brief Parse and check command line arguments
 * @param int argc - count of args
 * @param char *argv[] - array values of args
 * @return int - status 1(success) or -1(fail)
 */
int ParseAndCheckCommandLine(int argc, char* argv[]) {
    int opt = 0;
    int help = 0;
    char* string = "hi:m:d:c:l:g:";

    printf("%sParsing input parameters\n", info);

    while ((opt = getopt(argc, argv, string)) != -1) {
        switch (opt) {
        case 'h':
            showUsage();
            help = 1;
            break;
        case 'i':
            img_msg = optarg;
            break;
        case 'm':
            input_model = optarg;
            break;
        case 'd':
            device_name = optarg;
            break;
        case 'c':
            custom_plugin_cfg_msg = optarg;
            break;
        case 'l':
            custom_ex_library_msg = optarg;
            break;
        case 'g':
            config_msg = optarg;
            break;
        default:
            return -1;
        }
    }

    if (help)
        return -1;
    if (input_model == NULL) {
        printf("Model is required but not set. Please set -m option.\n");
        return -1;
    }
    if (img_msg == NULL) {
        printf("Input is required but not set.Please set -i option.\n");
        return -1;
    }

    return 1;
}

/**
 * @brief This function checks input args and existence of specified files in a
 * given folder. Updated the file_paths and file_num.
 * @param arg path to a file to be checked for existence
 * @return none.
 */
void readInputFilesArgument(const char* arg) {
    struct stat sb;
    int i;
    if (stat(arg, &sb) != 0) {
        printf("%sFile %s cannot be opened!\n", warn, arg);
        return;
    }
    if (S_ISDIR(sb.st_mode)) {
        DIR* dp;
        dp = opendir(arg);
        if (dp == NULL) {
            printf("%sFile %s cannot be opened!\n", warn, arg);
            return;
        }

        struct dirent* ep;
        while (NULL != (ep = readdir(dp))) {
            const char* fileName = ep->d_name;
            if (strcmp(fileName, ".") == 0 || strcmp(fileName, "..") == 0)
                continue;
            char* file_path = (char*)calloc(strlen(arg) + strlen(ep->d_name) + 2, sizeof(char));
            memcpy(file_path, arg, strlen(arg));
            memcpy(file_path + strlen(arg), "/", strlen("/"));
            memcpy(file_path + strlen(arg) + strlen("/"), ep->d_name, strlen(ep->d_name) + 1);

            if (file_num == 0) {
                file_paths = (char**)calloc(1, sizeof(char*));
                file_paths[0] = file_path;
                ++file_num;
            } else {
                char** temp = (char**)realloc(file_paths, sizeof(char*) * (file_num + 1));
                if (temp) {
                    file_paths = temp;
                    file_paths[file_num++] = file_path;
                } else {
                    for (i = 0; i < file_num; ++i) {
                        free(file_paths[i]);
                    }
                    free(file_path);
                    free(file_paths);
                    file_num = 0;
                }
            }
        }
        closedir(dp);
        dp = NULL;
    } else {
        char* file_path = (char*)calloc(strlen(arg) + 1, sizeof(char));
        memcpy(file_path, arg, strlen(arg) + 1);
        if (file_num == 0) {
            file_paths = (char**)calloc(1, sizeof(char*));
        }
        file_paths[file_num++] = file_path;
    }
}

/**
 * @brief This function find -i key in input args. It's necessary to process
 * multiple values for single key
 * @return none.
 */
void parseInputFilesArguments(int argc, char** argv) {
    int readArguments = 0, i;
    for (i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0) {
            readArguments = 1;
            continue;
        }
        if (!readArguments) {
            continue;
        }
        if (argv[i][0] == '-') {
            break;
        }
        readInputFilesArgument(argv[i]);
    }

    if (file_num < MAX_IMAGES) {
        printf("%sFiles were added: %d\n", info, file_num);
        for (i = 0; i < file_num; ++i) {
            printf("%s    %s\n", info, file_paths[i]);
        }
    } else {
        printf("%sFiles were added: %d. Too many to display each of them.\n", info, file_num);
    }
}

/**
 * @brief Convert the contents of configuration file to the ie_config_t struct.
 * @param config_file File path.
 * @param comment Separator symbol.
 * @return A pointer to the ie_config_t instance.
 */
ie_config_t* parseConfig(const char* config_file, char comment) {
    FILE* file = fopen(config_file, "r");
    if (!file) {
        return NULL;
    }

    ie_config_t* cfg = NULL;
    char key[256], value[256];

    if (fscanf(file, "%s", key) != EOF && fscanf(file, "%s", value) != EOF) {
        char* cfg_name = (char*)calloc(strlen(key) + 1, sizeof(char));
        char* cfg_value = (char*)calloc(strlen(value) + 1, sizeof(char));
        memcpy(cfg_name, key, strlen(key) + 1);
        memcpy(cfg_value, value, strlen(value) + 1);
        ie_config_t* cfg_t = (ie_config_t*)calloc(1, sizeof(ie_config_t));
        cfg_t->name = cfg_name;
        cfg_t->value = cfg_value;
        cfg_t->next = NULL;
        cfg = cfg_t;
    }
    if (cfg) {
        ie_config_t* cfg_temp = cfg;
        while (fscanf(file, "%s", key) != EOF && fscanf(file, "%s", value) != EOF) {
            if (strlen(key) == 0 || key[0] == comment) {
                continue;
            }
            char* cfg_name = (char*)calloc(strlen(key) + 1, sizeof(char));
            char* cfg_value = (char*)calloc(strlen(value) + 1, sizeof(char));
            memcpy(cfg_name, key, strlen(key) + 1);
            memcpy(cfg_value, value, strlen(value) + 1);
            ie_config_t* cfg_t = (ie_config_t*)calloc(1, sizeof(ie_config_t));
            cfg_t->name = cfg_name;
            cfg_t->value = cfg_value;
            cfg_t->next = NULL;
            cfg_temp->next = cfg_t;
            cfg_temp = cfg_temp->next;
        }
    }
    fclose(file);

    return cfg;
}

/**
 * @brief Releases memory occupied by config
 * @param config A pointer to the config to free memory.
 * @return none
 */
void config_free(ie_config_t* config) {
    while (config) {
        ie_config_t* temp = config;
        if (config->name) {
            free((char*)config->name);
            config->name = NULL;
        }
        if (config->value) {
            free((char*)config->value);
            config->value = NULL;
        }
        if (config->next) {
            config = config->next;
        }

        free(temp);
        temp = NULL;
    }
}

/**
 * @brief Convert the numbers to char *;
 * @param str A pointer to the converted string.
 * @param num The number to convert.
 * @return none.
 */
void int2str(char* str, int num) {
    int i = 0, j;
    if (num == 0) {
        str[0] = '0';
        str[1] = '\0';
        return;
    }

    while (num != 0) {
        str[i++] = num % 10 + '0';
        num = num / 10;
    }

    str[i] = '\0';
    --i;
    for (j = 0; j < i; ++j, --i) {
        char temp = str[j];
        str[j] = str[i];
        str[i] = temp;
    }
}

int main(int argc, char** argv) {
    /** This sample covers certain topology and cannot be generalized for any
     * object detection one **/
    // ------------------------------ Get Inference Engine API version
    // ---------------------------------
    ie_version_t version = ie_c_api_version();
    printf("%sInferenceEngine: \n", info);
    printf("%s\n", version.api_version);
    ie_version_free(&version);

    // ------------------------------ Parsing and validation of input args
    // ---------------------------------

    char** argv_temp = (char**)calloc(argc, sizeof(char*));
    if (!argv_temp) {
        return EXIT_FAILURE;
    }

    int i, j;
    for (i = 0; i < argc; ++i) {
        argv_temp[i] = argv[i];
    }

    char *input_weight = NULL, *imageInputName = NULL, *imInfoInputName = NULL, *output_name = NULL;
    ie_core_t* core = NULL;
    ie_network_t* network = NULL;
    ie_executable_network_t* exe_network = NULL;
    ie_infer_request_t* infer_request = NULL;
    ie_blob_t *imageInput = NULL, *output_blob = NULL;

    if (ParseAndCheckCommandLine(argc, argv) < 0) {
        free(argv_temp);
        return EXIT_FAILURE;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Read input
    // -----------------------------------------------------------
    /** This file_paths stores paths to the processed images **/
    parseInputFilesArguments(argc, argv_temp);
    if (!file_num) {
        printf("No suitable images were found\n");
        free(argv_temp);
        return EXIT_FAILURE;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 1. Initialize inference engine core
    // -------------------------------------

    printf("%sLoading Inference Engine\n", info);
    IEStatusCode status = ie_core_create("", &core);
    if (status != OK)
        goto err;

    // ------------------------------ Get Available Devices
    // ------------------------------------------------------
    ie_core_versions_t ver;
    printf("%sDevice info: \n", info);
    status = ie_core_get_versions(core, device_name, &ver);
    if (status != OK)
        goto err;
    for (i = 0; i < ver.num_vers; ++i) {
        printf("         %s\n", ver.versions[i].device_name);
        printf("         %s version ......... %zu.%zu\n", ver.versions[i].description, ver.versions[i].major, ver.versions[i].minor);
        printf("         Build ......... %s\n", ver.versions[i].build_number);
    }
    ie_core_versions_free(&ver);

    if (custom_ex_library_msg) {
        // Custom CPU extension is loaded as a shared library and passed as a
        // pointer to base extension
        status = ie_core_add_extension(core, custom_ex_library_msg, "CPU");
        if (status != OK)
            goto err;
        printf("%sCustom extension loaded: %s\n", info, custom_ex_library_msg);
    }

    if (custom_plugin_cfg_msg && (strcmp(device_name, "GPU") == 0 || strcmp(device_name, "MYRIAD") == 0 || strcmp(device_name, "HDDL") == 0)) {
        // Config for device plugin custom extension is loaded from an .xml
        // description
        ie_config_t cfg = {"CONFIG_FILE", custom_plugin_cfg_msg, NULL};
        status = ie_core_set_config(core, &cfg, device_name);
        if (status != OK)
            goto err;
        printf("%sConfig for device plugin custom extension loaded: %s\n", info, custom_plugin_cfg_msg);
    }
    // -----------------------------------------------------------------------------------------------------

    // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin
    // files) or ONNX (.onnx file) format
    printf("%sLoading network:\n", info);
    printf("\t%s\n", input_model);
    status = ie_core_read_network(core, input_model, NULL, &network);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 3. Configure input & output
    // ---------------------------------------------
    // --------------------------- Prepare input blobs
    // -----------------------------------------------------
    printf("%sPreparing input blobs\n", info);

    /** SSD network has one input and one output **/
    size_t input_num = 0;
    status = ie_network_get_inputs_number(network, &input_num);
    if (status != OK || (input_num != 1 && input_num != 2)) {
        printf("Sample supports topologies only with 1 or 2 inputs\n");
        goto err;
    }

    /**
     * Some networks have SSD-like output format (ending with DetectionOutput
     * layer), but having 2 inputs as Faster-RCNN: one for image and one for
     * "image info".
     *
     * Although object_datection_sample_ssd's main task is to support clean SSD,
     * it could score the networks with two inputs as well. For such networks
     * imInfoInputName will contain the "second" input name.
     */
    size_t input_width = 0, input_height = 0;

    /** Stores input image **/

    /** Iterating over all input blobs **/
    for (i = 0; i < input_num; ++i) {
        char* name = NULL;
        status |= ie_network_get_input_name(network, i, &name);
        dimensions_t input_dim;
        status |= ie_network_get_input_dims(network, name, &input_dim);
        if (status != OK)
            goto err;

        /** Working with first input tensor that stores image **/
        if (input_dim.ranks == 4) {
            imageInputName = name;
            input_height = input_dim.dims[2];
            input_width = input_dim.dims[3];

            /** Creating first input blob **/
            status = ie_network_set_input_precision(network, name, U8);
            if (status != OK)
                goto err;
        } else if (input_dim.ranks == 2) {
            imInfoInputName = name;

            status = ie_network_set_input_precision(network, name, FP32);
            if (status != OK || (input_dim.dims[1] != 3 && input_dim.dims[1] != 6)) {
                printf("Invalid input info. Should be 3 or 6 values length\n");
                goto err;
            }
        }
    }

    if (imageInputName == NULL) {
        status = ie_network_get_input_name(network, 0, &imageInputName);
        if (status != OK)
            goto err;
        dimensions_t input_dim;
        status = ie_network_get_input_dims(network, imageInputName, &input_dim);
        if (status != OK)
            goto err;
        input_height = input_dim.dims[2];
        input_width = input_dim.dims[3];
    }

    /** Collect images data **/
    c_mat_t* originalImages = (c_mat_t*)calloc(file_num, sizeof(c_mat_t));
    c_mat_t* images = (c_mat_t*)calloc(file_num, sizeof(c_mat_t));

    if (!originalImages || !images)
        goto err;

    int image_num = 0;
    for (i = 0; i < file_num; ++i) {
        c_mat_t img = {NULL, 0, 0, 0, 0, 0};
        if (image_read(file_paths[i], &img) == -1) {
            printf("%sImage %s cannot be read!\n", warn, file_paths[i]);
            continue;
        }
        /** Store image data **/
        c_mat_t resized_img = {NULL, 0, 0, 0, 0, 0};
        if ((input_width == img.mat_width) && (input_height == img.mat_height)) {
            resized_img.mat_data_size = img.mat_data_size;
            resized_img.mat_channels = img.mat_channels;
            resized_img.mat_width = img.mat_width;
            resized_img.mat_height = img.mat_height;
            resized_img.mat_type = img.mat_type;
            resized_img.mat_data = calloc(1, resized_img.mat_data_size);
            if (resized_img.mat_data == NULL) {
                image_free(&img);
                continue;
            }

            for (j = 0; j < resized_img.mat_data_size; ++j)
                resized_img.mat_data[j] = img.mat_data[j];
        } else {
            printf("%sImage is resized from (%d, %d) to (%zu, %zu)\n", warn, img.mat_width, img.mat_height, input_width, input_height);

            if (image_resize(&img, &resized_img, (int)input_width, (int)input_height) == -1) {
                printf("%sImage %s cannot be resized!\n", warn, file_paths[i]);
                image_free(&img);
                continue;
            }
        }

        originalImages[image_num] = img;
        images[image_num] = resized_img;
        ++image_num;
    }

    if (!image_num) {
        printf("Valid input images were not found!\n");
        free(originalImages);
        free(images);
        goto err;
    }

    input_shapes_t shapes;
    status = ie_network_get_input_shapes(network, &shapes);
    if (status != OK)
        goto err;

    /** Using ie_network_reshape() to set the batch size equal to the number of
     * input images **/
    /** For input with NCHW/NHWC layout the first dimension N is the batch size
     * **/
    shapes.shapes[0].shape.dims[0] = image_num;
    status = ie_network_reshape(network, shapes);
    if (status != OK)
        goto err;
    ie_network_input_shapes_free(&shapes);

    input_shapes_t shapes2;
    status = ie_network_get_input_shapes(network, &shapes2);
    if (status != OK)
        goto err;
    size_t batchSize = shapes2.shapes[0].shape.dims[0];
    ie_network_input_shapes_free(&shapes2);
    printf("%sBatch size is %zu\n", info, batchSize);

    // --------------------------- Prepare output blobs
    // ----------------------------------------------------
    printf("%sPreparing output blobs\n", info);

    size_t output_num = 0;
    status = ie_network_get_outputs_number(network, &output_num);

    if (status != OK || !output_num) {
        printf("Can't find a DetectionOutput layer in the topology\n");
        goto err;
    }

    status = ie_network_get_output_name(network, output_num - 1, &output_name);
    if (status != OK)
        goto err;

    dimensions_t output_dim;
    status = ie_network_get_output_dims(network, output_name, &output_dim);
    if (status != OK)
        goto err;
    if (output_dim.ranks != 4) {
        printf("Incorrect output dimensions for SSD model\n");
        goto err;
    }

    const int maxProposalCount = (int)output_dim.dims[2];
    const int objectSize = (int)output_dim.dims[3];

    if (objectSize != 7) {
        printf("Output item should have 7 as a last dimension\n");
        goto err;
    }

    /** Set the precision of output data provided by the user, should be called
     * before load of the network to the device **/
    status = ie_network_set_output_precision(network, output_name, FP32);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 4. Loading model to the device
    // ------------------------------------------
    printf("%sLoading model to the device\n", info);
    if (config_msg) {
        ie_config_t* config = parseConfig(config_msg, '#');
        status = ie_core_load_network(core, network, device_name, config, &exe_network);
        config_free(config);
        if (status != OK) {
            goto err;
        }
    } else {
        ie_config_t cfg = {NULL, NULL, NULL};
        status = ie_core_load_network(core, network, device_name, &cfg, &exe_network);
        if (status != OK)
            goto err;
    }

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 5. Create infer request
    // -------------------------------------------------
    printf("%sCreate infer request\n", info);
    status = ie_exec_network_create_infer_request(exe_network, &infer_request);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 6. Prepare input
    // --------------------------------------------------------

    /** Creating input blob **/
    status = ie_infer_request_get_blob(infer_request, imageInputName, &imageInput);
    if (status != OK)
        goto err;

    /** Filling input tensor with images. First b channel, then g and r channels
     * **/
    dimensions_t input_tensor_dims;
    status = ie_blob_get_dims(imageInput, &input_tensor_dims);
    if (status != OK)
        goto err;
    size_t num_channels = input_tensor_dims.dims[1];
    size_t image_size = input_tensor_dims.dims[3] * input_tensor_dims.dims[2];

    ie_blob_buffer_t blob_buffer;
    status = ie_blob_get_buffer(imageInput, &blob_buffer);
    if (status != OK)
        goto err;
    unsigned char* data = (unsigned char*)(blob_buffer.buffer);

    /** Iterate over all input images **/
    int image_id, pid, ch, k;
    for (image_id = 0; image_id < batchSize; ++image_id) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (pid = 0; pid < image_size; ++pid) {
            /** Iterate over all channels **/
            for (ch = 0; ch < num_channels; ++ch) {
                /**          [images stride + channels stride + pixel id ] all in bytes
                 * **/
                data[image_id * image_size * num_channels + ch * image_size + pid] = images[image_id].mat_data[pid * num_channels + ch];
            }
        }
        image_free(&images[image_id]);
    }
    free(images);
    ie_blob_free(&imageInput);

    if (imInfoInputName != NULL) {
        ie_blob_t* input2 = NULL;
        status = ie_infer_request_get_blob(infer_request, imInfoInputName, &input2);

        dimensions_t imInfoDim;
        status |= ie_blob_get_dims(input2, &imInfoDim);
        // Fill input tensor with values
        ie_blob_buffer_t info_blob_buffer;
        status |= ie_blob_get_buffer(input2, &info_blob_buffer);
        if (status != OK) {
            ie_blob_free(&input2);
            goto err;
        }
        float* p = (float*)(info_blob_buffer.buffer);
        for (image_id = 0; image_id < batchSize; ++image_id) {
            p[image_id * imInfoDim.dims[1] + 0] = (float)input_height;
            p[image_id * imInfoDim.dims[1] + 1] = (float)input_width;

            for (k = 2; k < imInfoDim.dims[1]; k++) {
                p[image_id * imInfoDim.dims[1] + k] = 1.0f;  // all scale factors are set to 1.0
            }
        }
        ie_blob_free(&input2);
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 7. Do inference
    // --------------------------------------------------------
    printf("%sStart inference\n", info);
    status = ie_infer_request_infer_async(infer_request);
    status |= ie_infer_request_wait(infer_request, -1);
    if (status != OK)
        goto err;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 8. Process output
    // ------------------------------------------------------
    printf("%sProcessing output blobs\n", info);

    status = ie_infer_request_get_blob(infer_request, output_name, &output_blob);
    if (status != OK)
        goto err;

    ie_blob_buffer_t output_blob_buffer;
    status = ie_blob_get_cbuffer(output_blob, &output_blob_buffer);
    if (status != OK)
        goto err;
    const float* detection = (float*)(output_blob_buffer.cbuffer);

    int** classes = (int**)calloc(image_num, sizeof(int*));
    rectangle_t** boxes = (rectangle_t**)calloc(image_num, sizeof(rectangle_t*));
    int* object_num = (int*)calloc(image_num, sizeof(int));
    for (i = 0; i < image_num; ++i) {
        classes[i] = (int*)calloc(maxProposalCount, sizeof(int));
        boxes[i] = (rectangle_t*)calloc(maxProposalCount, sizeof(rectangle_t));
        object_num[i] = 0;
    }

    /* Each detection has image_id that denotes processed image */
    int curProposal;
    for (curProposal = 0; curProposal < maxProposalCount; curProposal++) {
        image_id = (int)(detection[curProposal * objectSize + 0]);
        if (image_id < 0) {
            break;
        }

        float confidence = detection[curProposal * objectSize + 2];
        int label = (int)(detection[curProposal * objectSize + 1]);
        int xmin = (int)(detection[curProposal * objectSize + 3] * originalImages[image_id].mat_width);
        int ymin = (int)(detection[curProposal * objectSize + 4] * originalImages[image_id].mat_height);
        int xmax = (int)(detection[curProposal * objectSize + 5] * originalImages[image_id].mat_width);
        int ymax = (int)(detection[curProposal * objectSize + 6] * originalImages[image_id].mat_height);

        printf("[%d, %d] element, prob = %f    (%d, %d)-(%d, %d) batch id : %d", curProposal, label, confidence, xmin, ymin, xmax, ymax, image_id);

        if (confidence > 0.5) {
            /** Drawing only objects with >50% probability **/
            classes[image_id][object_num[image_id]] = label;
            boxes[image_id][object_num[image_id]].x_min = xmin;
            boxes[image_id][object_num[image_id]].y_min = ymin;
            boxes[image_id][object_num[image_id]].rect_width = xmax - xmin;
            boxes[image_id][object_num[image_id]].rect_height = ymax - ymin;
            printf(" WILL BE PRINTED!");
            ++object_num[image_id];
        }
        printf("\n");
    }
    /** Adds rectangles to the image and save **/
    int batch_id;
    for (batch_id = 0; batch_id < batchSize; ++batch_id) {
        if (object_num[batch_id] > 0) {
            image_add_rectangles(&originalImages[batch_id], boxes[batch_id], classes[batch_id], object_num[batch_id], 2);
        }
        const char* out = "out_";
        char str_num[16] = {0};
        int2str(str_num, batch_id);
        char* img_path = (char*)calloc(strlen(out) + strlen(str_num) + strlen(".bmp") + 1, sizeof(char));
        memcpy(img_path, out, strlen(out));
        memcpy(img_path + strlen(out), str_num, strlen(str_num));
        memcpy(img_path + strlen(out) + strlen(str_num), ".bmp", strlen(".bmp") + 1);
        image_save(img_path, &originalImages[batch_id]);
        printf("%sImage %s created!\n", info, img_path);
        free(img_path);
        image_free(&originalImages[batch_id]);
    }
    free(originalImages);
    // -----------------------------------------------------------------------------------------------------

    printf("%sExecution successful\n", info);
    printf("\nThis sample is an API example,"
           " for any performance measurements please use the dedicated benchmark_"
           "app tool\n");

    for (i = 0; i < image_num; ++i) {
        free(classes[i]);
        free(boxes[i]);
    }
    free(classes);
    free(boxes);
    free(object_num);
    ie_blob_free(&output_blob);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
    ie_network_name_free(&imageInputName);
    ie_network_name_free(&imInfoInputName);
    ie_network_name_free(&output_name);
    free(input_weight);
    free(argv_temp);

    return EXIT_SUCCESS;
err:
    free(argv_temp);
    if (input_weight)
        free(input_weight);
    if (core)
        ie_core_free(&core);
    if (network)
        ie_network_free(&network);
    if (imageInputName)
        ie_network_name_free(&imageInputName);
    if (imInfoInputName)
        ie_network_name_free(&imInfoInputName);
    if (output_name)
        ie_network_name_free(&output_name);
    if (exe_network)
        ie_exec_network_free(&exe_network);
    if (imageInput)
        ie_blob_free(&imageInput);
    if (output_blob)
        ie_blob_free(&output_blob);

    return EXIT_FAILURE;
}
