// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <c_api/ie_c_api.h>

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

using namespace std;
class PluginLoader {
public:
    explicit PluginLoader(const std::string& libraryPath);
    ~PluginLoader();

    template <typename FunctionType>
    std::function<FunctionType> getEntry(const std::string& funcName);

private:
    void* retrieveFunc(const std::string& funcName);
#ifdef _WIN32
    HMODULE m_handler;
#else
    void* m_handler{nullptr};
#endif
    std::string m_name;
};

template <typename FunctionType>
std::function<FunctionType> PluginLoader::getEntry(const std::string& funcName) {
    void* func = retrieveFunc(funcName);
    return reinterpret_cast<FunctionType*>(func);
}

#ifdef _WIN32

PluginLoader::PluginLoader(const std::string& libraryPath) {
    m_name = libraryPath;
    m_handler = LoadLibraryA(libraryPath.c_str());
    if (!m_handler) {
        std::cout << "LoadLibrary " << libraryPath.c_str() << " ,error: " << GetLastError() << std::endl;
        throw std::system_error(GetLastError(), std::generic_category(), "load library failed.");
    }
}

PluginLoader::~PluginLoader() {
    FreeLibrary(m_handler);
    std::cout << "FreeLibrary " << m_name.c_str() << " ,error = " << GetLastError() << std::endl;
    m_handler = nullptr;
}

void* PluginLoader::retrieveFunc(const std::string& funcName) {
    void* func = GetProcAddress(m_handler, funcName.c_str());
    if (!func) {
        std::cout << "extract symbol " << funcName.c_str() << " ,error: " << GetLastError() << std::endl;
        throw std::system_error(GetLastError(), std::generic_category(), "extract symbol error.");
    }
    return func;
}

#else

PluginLoader::PluginLoader(const std::string& libraryPath) {
    m_name = libraryPath;
    m_handler = dlopen(libraryPath.c_str(), RTLD_LAZY);
    if (!m_handler) {
        std::cout << "dlopen " << libraryPath.c_str() << " ,error: " << dlerror() << std::endl;
        throw std::system_error(ELIBACC, std::generic_category(), "load library failed.");
    }
}

PluginLoader::~PluginLoader() {
    dlclose(m_handler);
    std::cout << "dlclose " << m_name.c_str() << std::endl;
    m_handler = nullptr;
}

void* PluginLoader::retrieveFunc(const std::string& funcName) {
    void* func = dlsym(m_handler, funcName.c_str());
    if (!func) {
        std::cout << "extract symbol " << funcName.c_str() << " ,error: " << dlerror() << std::endl;
        throw std::system_error(ELIBBAD, std::generic_category(), "extract symbol error.");
    } else {
        std::cout << "extract symbol " << funcName.c_str() << "..." << std::endl;
    }
    return func;
}

#endif

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

int main(int argc, char** argv) {
    // ------------------------------ Parsing and validation of input args
    // ---------------------------------
    if (argc < 4) {
        std::cout << "Format: runtime_load <openvino_c lib full path> <path_to_model> <device_name>" << std::endl;
        std::cout << "Please assign lib or dll full path name as input." << std::endl;
        return 0;
    }

    const char* input_model = argv[2];
    const char* input_image_path = "random";
    const char* device_name = argv[3];
    ie_core_t* core = NULL;
    ie_network_t* network = NULL;
    ie_executable_network_t* exe_network = NULL;
    ie_infer_request_t* infer_request = NULL;
    char *input_name = NULL, *output_name = NULL;
    ie_blob_t *imgBlob = NULL, *output_blob = NULL;
    size_t network_input_size;
    size_t network_output_size;
    // -----------------------------------------------------------------------------------------------------

    // ------------------------------ load library and get its function entries
    // ---------------------------------
    std::string libFullName(argv[1]);
    std::shared_ptr<PluginLoader> loader = std::make_shared<PluginLoader>(libFullName);
    std::cout << "Load lib: " << libFullName.c_str() << std::endl;

    using ie_core_create_f = IEStatusCode(const char*, ie_core_t**);
    using ie_core_read_network_f = IEStatusCode(ie_core_t*, const char*, const char*, ie_network_t**);
    using ie_network_get_inputs_number_f = IEStatusCode(const ie_network_t*, size_t*);
    using ie_network_get_outputs_number_f = IEStatusCode(const ie_network_t*, size_t*);
    using ie_network_get_input_name_f = IEStatusCode(const ie_network_t*, size_t, char**);
    using ie_network_set_input_resize_algorithm_f = IEStatusCode(ie_network_t*, const char*, const resize_alg_e);
    using ie_network_set_input_layout_f = IEStatusCode(ie_network_t*, const char*, const layout_e);
    using ie_network_set_input_precision_f = IEStatusCode(ie_network_t*, const char*, const precision_e);
    using ie_network_get_output_name_f = IEStatusCode(const ie_network_t*, const size_t, char**);
    using ie_network_set_output_precision_f = IEStatusCode(ie_network_t*, const char*, const precision_e);
    using ie_core_load_network_f =
        IEStatusCode(ie_core_t*, const ie_network_t*, const char*, const ie_config_t*, ie_executable_network_t**);
    using ie_exec_network_create_infer_request_f = IEStatusCode(ie_executable_network_t*, ie_infer_request_t**);
    using ie_blob_make_memory_from_preallocated_f = IEStatusCode(const tensor_desc_t*, void*, size_t, ie_blob_t**);
    using ie_infer_request_set_blob_f = IEStatusCode(ie_infer_request_t*, const char*, const ie_blob_t*);
    using ie_infer_request_infer_f = IEStatusCode(ie_infer_request_t*);
    using ie_infer_request_get_blob_f = IEStatusCode(ie_infer_request_t*, const char*, ie_blob_t**);
    using ie_blob_get_dims_f = IEStatusCode(const ie_blob_t*, dimensions_t*);
    using ie_blob_get_cbuffer_f = IEStatusCode(const ie_blob_t*, ie_blob_buffer_t*);
    using ie_blob_free_f = void(ie_blob_t**);
    using ie_infer_request_free_f = void(ie_infer_request_t**);
    using ie_network_name_free_f = void(char**);
    using ie_exec_network_free_f = void(ie_executable_network_t**);
    using ie_network_free_f = void(ie_network_t**);
    using ie_core_free_f = void(ie_core_t**);
    using ie_core_set_config_f = IEStatusCode(ie_core_t*, const ie_config_t*, const char*);
    using ie_cleanup_f = void();

    // auto stub_ie_core_create = loader->getEntry<ie_core_create_f>("ie_core_create");
#define EXTRACT_FUNC(name) auto name = loader->getEntry<name##_f>(#name);

    EXTRACT_FUNC(ie_core_create)
    EXTRACT_FUNC(ie_core_read_network)
    EXTRACT_FUNC(ie_network_get_inputs_number)
    EXTRACT_FUNC(ie_network_get_outputs_number)
    EXTRACT_FUNC(ie_network_get_input_name)
    EXTRACT_FUNC(ie_network_set_input_resize_algorithm)
    EXTRACT_FUNC(ie_network_set_input_layout)
    EXTRACT_FUNC(ie_network_set_input_precision)
    EXTRACT_FUNC(ie_network_get_output_name)
    EXTRACT_FUNC(ie_network_set_output_precision)
    EXTRACT_FUNC(ie_core_load_network)
    EXTRACT_FUNC(ie_exec_network_create_infer_request)
    EXTRACT_FUNC(ie_blob_make_memory_from_preallocated)
    EXTRACT_FUNC(ie_infer_request_set_blob)
    EXTRACT_FUNC(ie_infer_request_infer)
    EXTRACT_FUNC(ie_infer_request_get_blob)
    EXTRACT_FUNC(ie_blob_get_dims)
    EXTRACT_FUNC(ie_blob_get_cbuffer)
    EXTRACT_FUNC(ie_blob_free)
    EXTRACT_FUNC(ie_infer_request_free)
    EXTRACT_FUNC(ie_exec_network_free)
    EXTRACT_FUNC(ie_network_name_free)
    EXTRACT_FUNC(ie_network_free)
    EXTRACT_FUNC(ie_core_free)
    EXTRACT_FUNC(ie_core_set_config)
    EXTRACT_FUNC(ie_cleanup)
#undef EXTRAC_FUNC

    // --------------------------- Step 1. Initialize inference engine core
    // -------------------------------------
    IEStatusCode status = ie_core_create("", &core);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_core_create status %d, line %d\n", status, __LINE__);
        return 1;
    }

    ie_config_t config_tbb = {"FORCE_TBB_TERMINATE", "YES", NULL};
    status = ie_core_set_config(core, &config_tbb, device_name);

    // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin
    // files) or ONNX (.onnx file) format
    status = ie_core_read_network(core, input_model, NULL, &network);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_core_read_network status %d, line %d\n", status, __LINE__);
        return 1;
    }
    // check the network topology
    status = ie_network_get_inputs_number(network, &network_input_size);
    if (status != OK || network_input_size != 1) {
        printf("Sample supports topologies with 1 input only\n");
        return 1;
    }

    status = ie_network_get_outputs_number(network, &network_output_size);
    if (status != OK || network_output_size != 1) {
        fprintf(stderr, "Sample supports topologies with 1 output only\n");
        return 1;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 3. Configure input & output
    // ---------------------------------------------
    // --------------------------- Prepare input blobs
    // -----------------------------------------------------

    status = ie_network_get_input_name(network, 0, &input_name);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_network_get_input_name status %d, line %d\n", status, __LINE__);
        return 1;
    }
    /* Mark input as resizable by setting of a resize algorithm.
     * In this case we will be able to set an input blob of any shape to an infer
     * request. Resize and layout conversions are executed automatically during
     * inference */
    auto status1 = ie_network_set_input_resize_algorithm(network, input_name, RESIZE_BILINEAR);
    auto status2 = ie_network_set_input_layout(network, input_name, NHWC);
    auto status3 = ie_network_set_input_precision(network, input_name, U8);
    if (status1 || status2 || status3) {
        fprintf(stderr, "ERROR ie_network_set_input_* status %d, line %d\n", status, __LINE__);
        return 1;
    }

    // --------------------------- Prepare output blobs
    // ----------------------------------------------------
    status1 = ie_network_get_output_name(network, 0, &output_name);
    status2 = ie_network_set_output_precision(network, output_name, FP32);
    if (status1 || status2) {
        fprintf(stderr, "ERROR ie_network_get_output_* status %d, line %d\n", status, __LINE__);
        return 1;
    }

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 4. Loading model to the device
    // ------------------------------------------
    ie_config_t config = {NULL, NULL, NULL};
    status = ie_core_load_network(core, network, device_name, &config, &exe_network);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_core_load_network status %d, line %d\n", status, __LINE__);
        return 1;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 5. Create infer request
    // -------------------------------------------------
    status = ie_exec_network_create_infer_request(exe_network, &infer_request);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_exec_network_create_infer_request status %d, line %d\n", status, __LINE__);
        return 1;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 6. Prepare input
    // --------------------------------------------------------
    /* Read input image to a blob and set it to an infer request without resize
     * and layout conversions. */
    size_t size = 4 * 3 * 800 * 600;
    dimensions_t dimens = {4, {1, 3, 800, 600}};
    tensor_desc_t tensorDesc = {NCHW, dimens, U8};
    char* temp_data = (char*)malloc(size);

    status = ie_blob_make_memory_from_preallocated(&tensorDesc, temp_data, size, &imgBlob);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_blob_make_memory_from_preallocated status %d, line %d\n", status, __LINE__);
        free(temp_data);
        return 1;
    }
    // infer_request accepts input blob of any size

    status = ie_infer_request_set_blob(infer_request, input_name, imgBlob);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_infer_request_set_blob status %d, line %d\n", status, __LINE__);
        return 1;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 7. Do inference
    // --------------------------------------------------------
    /* Running the request synchronously */
    status = ie_infer_request_infer(infer_request);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_infer_request_infer status %d, line %d\n", status, __LINE__);
        return 1;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- Step 8. Process output
    // ------------------------------------------------------
    status = ie_infer_request_get_blob(infer_request, output_name, &output_blob);
    if (status != OK) {
        fprintf(stderr, "ERROR ie_infer_request_get_blob status %d, line %d\n", status, __LINE__);
        free(temp_data);
        return 1;
    }
    size_t class_num;
    // Convert output blob to classify struct for processing results
    // struct classify_res* cls = output_blob_to_classify_res(output_blob, &class_num);
    dimensions_t output_dim;
    status = ie_blob_get_dims(output_blob, &output_dim);
    if (status != OK) {
        std::cout << "Failed to ie_blob_get_dims!!! " << std::endl;
        return 1;
    }

    class_num = output_dim.dims[1];

    struct classify_res* cls = (struct classify_res*)malloc(sizeof(struct classify_res) * (class_num));
    if (!cls) {
        std::cout << "Failed to malloc classify_res!!! " << std::endl;
        return 1;
    }

    ie_blob_buffer_t blob_cbuffer;
    status = ie_blob_get_cbuffer(output_blob, &blob_cbuffer);
    if (status != OK) {
        std::cout << "Failed to ie_blob_get_cbuffer!!! " << std::endl;
        free(cls);
        return 1;
    }
    float* blob_data = (float*)(blob_cbuffer.cbuffer);

    size_t i;
    for (i = 0; i < class_num; ++i) {
        cls[i].class_id = i;
        cls[i].probability = blob_data[i];
    }

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
    ie_blob_free(&imgBlob);
    free(temp_data);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_name_free(&input_name);
    ie_network_name_free(&output_name);
    ie_network_free(&network);
    ie_core_free(&core);
    ie_cleanup();

    loader = nullptr;
    std::cout << "APP will exit after 10s" << std::endl << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "APP has exited" << std::endl;
    return EXIT_SUCCESS;
}