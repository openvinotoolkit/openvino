// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ov_c_api.h
 * C API of OpenVINO 2.0 bridge unlocks using of OpenVINO 2.0
 * library and all its plugins in native applications disabling usage
 * of C++ API. The scope of API covers significant part of C++ API and includes
 * an ability to read model from the disk, modify input and output information
 * to correspond their runtime representation like data types or memory layout,
 * load in-memory model to different devices including
 * heterogeneous and multi-device modes, manage memory where input and output
 * is allocated and manage inference flow.
**/

/**
 *  @defgroup ov_c_api OpenVINO 2.0 C API
 * OpenVINO 2.0 C API
 */

#ifndef OV_C_API_H
#define OV_C_API_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#ifdef __cplusplus
    #define OPENVINO_C_API_EXTERN extern "C"
#else
    #define OPENVINO_C_API_EXTERN
#endif

#if defined(OPENVINO_STATIC_LIBRARY) || defined(__GNUC__) && (__GNUC__ < 4)
    #define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __VA_ARGS__
    #define OV_NODISCARD
#else
    #if defined(_WIN32)
        #define OPENVINO_C_API_CALLBACK __cdecl
        #ifdef openvino_c_EXPORTS
            #define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __declspec(dllexport) __VA_ARGS__ __cdecl
        #else
            #define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __declspec(dllimport) __VA_ARGS__ __cdecl
        #endif
        #define OV_NODISCARD
    #else
        #define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __attribute__((visibility("default"))) __VA_ARGS__
        #define OV_NODISCARD __attribute__((warn_unused_result))
    #endif
#endif

#ifndef OPENVINO_C_API_CALLBACK
    #define OPENVINO_C_API_CALLBACK
#endif

/**
 * Max dimension of shape
 */
#define MAX_DIMENSION 8

/**
 * @struct ov_call_back_t
 * @brief Completion callback definition about the function and args
 */
typedef struct{
    void (OPENVINO_C_API_CALLBACK *callback_func)(void *args);
    void *args;
} ov_call_back_t;

/**
 * @struct ov_ProfilingInfo_t
 */
typedef struct{
    /**
     * @brief Defines the general status of a node.
     */
    enum Status {
        NOT_RUN,        //!< A node is not executed.
        OPTIMIZED_OUT,  //!< A node is optimized out during graph optimization phase.
        EXECUTED        //!< A node is executed.
    } status;

    /**
     * @brief The absolute time, in microseconds, that the node ran (in total).
     */
    int64_t real_time;

    /**
     * @brief The net host CPU time that the node ran.
     */
    int64_t cpu_time;

    /**
     * @brief Name of a node.
     */
    char* node_name;

    /**
     * @brief Execution type of a unit.
     */
    char* exec_type;

    /**
     * @brief Node type.
     */
    char* node_type;
}ov_profiling_info_t;

/**
 * @struct ov_profiling_info_list_t
 */
typedef struct {
    ov_profiling_info_t* profiling_infos;
    size_t num;
}ov_profiling_info_list_t;

/**
 * @enum ov_status_code_e
 * @brief This enum contains codes for all possible return values of the interface functions
 */
typedef enum {
    OK = 0,
    GENERAL_ERROR = 1,
    NOT_IMPLEMENTED = 2,
    NETWORK_NOT_LOADED = 3,
    PARAMETER_MISMATCH = 4,
    NOT_FOUND = 5,
    OUT_OF_BOUNDS = 6,
    /*
     * @brief exception not of std::exception derived type was thrown
     */
    UNEXPECTED = 7,
    REQUEST_BUSY = 8,
    RESULT_NOT_READY = 9,
    NOT_ALLOCATED = 10,
    INFER_NOT_STARTED = 11,
    NETWORK_NOT_READ = 12,
    INFER_CANCELLED = 13,
    UNKNOWN_ERROR = 14,
} ov_status_e;

/**
 * @struct ov_version_t
 */
typedef struct ov_version {
    char* buildNumber;
    char* description;
}ov_version_t;

typedef struct {
    char* device_name;
    char* buildNumber;
    char* description;
}ov_core_version_t;

/**
 * @struct ov_core_version_list_t
 */
typedef struct {
    ov_core_version_t *versions;
    size_t num_vers;
}ov_core_version_list_t;

/**
 * @struct ov_core_t
 */
typedef struct ov_core ov_core_t;

/**
 * @struct ov_node_t
 */
typedef struct ov_node ov_node_t;

/**
 * @struct ov_output_node_t
 */
typedef struct ov_output_node ov_output_node_t;

/**
 * @struct ov_output_node_list_t
 */
typedef struct{
    ov_output_node_t *output_nodes;
    size_t num;
} ov_output_node_list_t;

/**
 * @struct ov_model_t
 */
typedef struct ov_model ov_model_t;

/**
 * @struct ov_preprocess_t
 */
typedef struct ov_preprocess ov_preprocess_t;

/**
 * @struct ov_preprocess_input_info_t
 */
typedef struct ov_preprocess_input_info ov_preprocess_input_info_t;

/**
 * @struct ov_preprocess_input_tensor_info_t
 */
typedef struct ov_preprocess_input_tensor_info ov_preprocess_input_tensor_info_t;

/**
 * @struct ov_preprocess_output_info_t
 */
typedef struct ov_preprocess_output_info ov_preprocess_output_info_t;

/**
 * @struct ov_preprocess_output_tensor_info_t
 */
typedef struct ov_preprocess_output_tensor_info ov_preprocess_output_tensor_info_t;

/**
 * @struct ov_preprocess_input_model_info_t
 */
typedef struct ov_preprocess_input_model_info ov_preprocess_input_model_info_t;

/**
 * @struct ov_preprocess_input_process_steps_t
 */
typedef struct ov_preprocess_input_process_steps ov_preprocess_input_process_steps_t;

typedef enum {
    RESIZE_LINEAR,
    RESIZE_CUBIC,
    RESIZE_NEAREST
} ov_preprocess_resize_algorithm_e;

/**
 * @struct ov_compiled_model_t
 */
typedef struct ov_compiled_model ov_compiled_model_t;

/**
 * @struct ov_infer_request_t
 */
typedef struct ov_infer_request ov_infer_request_t;

/**
 * @struct ov_tensor_t
 */
typedef struct ov_tensor ov_tensor_t;

/**
 * @struct ov_element_type_e
 */
typedef enum {
    UNDEFINED = 0,  //!< Undefined element type
    DYNAMIC,    //!< Dynamic element type
    BOOLEAN,    //!< boolean element type
    BF16,       //!< bf16 element type
    F16,        //!< f16 element type
    F32,        //!< f32 element type
    F64,        //!< f64 element type
    I4,         //!< i4 element type
    I8,         //!< i8 element type
    I16,        //!< i16 element type
    I32,        //!< i32 element type
    I64,        //!< i64 element type
    U1,         //!< binary element type
    U4,         //!< u4 element type
    U8,         //!< u8 element type
    U16,        //!< u16 element type
    U32,        //!< u32 element type
    U64         //!< u64 element type
} ov_element_type_e;

/**
 * @struct ov_layout_t
 */
typedef char ov_layout_t[MAX_DIMENSION];

/**
 * @struct ov_shape_t
 */
typedef struct {
    int ranks;
    size_t dims[MAX_DIMENSION];
} ov_shape_t;

/**
 * @struct ov_partial_shape_t
 * brief Class representing a shape that may be partially or totally dynamic.
 *
 * A PartialShape may have:
 * Dynamic rank. (Informal notation: `?`)
 * Static rank, but dynamic dimensions on some or all axes.
 *     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
 * Static rank, and static dimensions on all axes.
 *     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
 *
 * An interface to make user can initialize ov_partial_shape_t
 */
typedef struct ov_partial_shape ov_partial_shape_t;

/**
 * @enum ov_performance_mode_e
 * @brief Enum to define possible performance mode hints
 * @brief This represents OpenVINO 2.0 ov::hint::PerformanceMode entity.
 *  It is same with enum class ov::hint::PerformanceMode as below:
 *
 *   enum class PerformanceMode {
 *   UNDEFINED = -1,             //!<  Undefined value, performance setting may vary from device to device
 *   LATENCY = 1,                //!<  Optimize for latency
 *   THROUGHPUT = 2,             //!<  Optimize for throughput
 *   CUMULATIVE_THROUGHPUT = 3,  //!< Optimize for cumulative throughput
 *   };
 *
 *  There also is a map in C implement to keep it aligned with C++ definition.
 */
typedef enum {
    UNDEFINED_MODE = -1,        //!<  Undefined value, performance setting may vary from device to device
    LATENCY = 1,                //!<  Optimize for latency
    THROUGHPUT = 2,             //!<  Optimize for throughput
    CUMULATIVE_THROUGHPUT = 3,  //!< Optimize for cumulative throughput
} ov_performance_mode_e;

/**
 * @struct ov_available_devices_t
 * @brief Represent all available devices.
 */
typedef struct {
    char **devices;
    size_t num_devices;
} ov_available_devices_t;

typedef enum {
    SUPPORTED_PROPERTIES = 0,           //  Read-only property to get a string list of supported read-only properties.
    AVAILABLE_DEVICES,                  //  Read-only property to get a list of available device IDs
    OPTIMAL_NUMBER_OF_INFER_REQUESTS,   //  Read-only property to get an unsigned integer value of optimal number of compiled model infer requests.
    RANGE_FOR_ASYNC_INFER_REQUESTS,     //  Read-only property to provide a hint for a range for number of async infer requests. If device supports streams, the metric provides range for number of IRs per stream.
    RANGE_FOR_STREAMS,                  //  Read-only property to provide information about a range for streams on platforms where streams are supported
    FULL_DEVICE_NAME,                   //  Read-only property to get a string value representing a full device name.
    OPTIMIZATION_CAPABILITIES,          //  Read-only property to get a string list of capabilities options per device.
    MODEL_CACHE,                        //  Read-write property to set/get the directory which will be used to store any data cached by plugins.
    NUM_STREAMS,                        //  Read-write property to set/get the number of executor logical partitions
    AFFINITY,                           //  Read-write property to set/get the name for setting CPU affinity per thread option.
    INFERENCE_NUM_THREADS,              //  Read-write property to set/get the maximum number of threads that can be used for inference tasks
    PERFORMANCE_HINT,                   //  Read-write property, it is high-level OpenVINO Performance Hints unlike low-level properties that are individual (per-device),
                                        //  the hints are something that every device accepts and turns into device-specific settings
                                        //  detail see ov_performance_mode_e to get its hint's key name
    NETWORK_NAME,                       //  Read-only property to get a name of name of a model
    INFERENCE_PRECISION_HINT,           //  Read-write property to set the hint for device to use specified precision for inference
    OPTIMAL_BATCH_SIZE,                 //  Read-only property to query information optimal batch size for the given device and the network
    MAX_BATCH_SIZE,                     //  Read-only property to get maximum batch size which does not cause performance degradation due to memory swap impact.
    PERFORMANCE_HINT_NUM_REQUESTS,      //  (Optional) property that backs the Performance Hints
                                        //  by giving additional information on how many inference requests the application will be keeping in flight
                                        //  usually this value comes from the actual use-case (e.g. number of video-cameras, or other sources of inputs)
} ov_property_key_e;

typedef union {
    uint32_t value_u;
    char value_s[320];
    ov_performance_mode_e value_performance_mode;
}ov_property_value;

typedef struct ov_property{
    ov_property_key_e key;
    ov_property_value value;
    struct ov_property* next;
}ov_property_t;

/**
 * @brief Initialize a partial shape.
 * @param str is the input partial info string
 *  Dynamic rank:
 *     Example: "?"
 *  Static rank, but dynamic dimensions on some or all axes.
 *     Examples: "{1,2,?,4}" or "{?,?,?}" or "{1,2,-1,4}""
 *  Static rank, and static dimensions on all axes.
 *     Examples: "{1,2,3,4}" or "{6}" or "{}""
 *
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_partial_shape_init(ov_partial_shape_t** partial_shape, const char* str);

/**
 * @brief Parse the partial shape to readable string.
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(const char*) ov_partial_shape_parse(ov_partial_shape_t* partial_shape);

/**
 * @brief Release partial shape.
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_partial_shape_free(ov_partial_shape_t* partial_shape);

/**
 * @brief Covert partial shape to static shape.
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_partial_shape_to_shape(ov_partial_shape_t* partial_shape, ov_shape_t* shape);

/**
 * @brief Print the error info.
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(const char*) ov_get_error_info(ov_status_e status);

/**
 * @brief Get version of OpenVINO.
 * @param ov_version_t a pointer to the version
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_get_version(ov_version_t *version);

/**
 * @brief Release the memory allocated by ov_version_t.
 * @param version A pointer to the ov_version_t to free memory.
 */
OPENVINO_C_API(void) ov_version_free(ov_version_t *version);

/**
 * @brief Constructs OpenVINO Core instance using XML configuration file with devices description.
 * See RegisterPlugins for more details. 
 * @param xml_config_file A path to .xml file with devices to load from. If XML configuration file is not specified,
 * then default plugin.xml file will be used.
 * @param core A pointer to the newly created ov_core_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_create(const char *xml_config_file, ov_core_t **core);

/**
 * @brief Release the memory allocated by ov_core_t.
 * @param core A pointer to the ov_core_t to free memory.
 */
OPENVINO_C_API(void) ov_core_free(ov_core_t *core);

/**
 * @brief Reads the model from the .xml and .bin files of the IR.
 * @param core A pointer to the ie_core_t instance.
 * @param model_path .xml file's path of the IR.
 * @param bin_path .bin file's path of the IR, if path is empty, will try to read bin file with the same name as xml and
 * if bin file with the same name was not found, will load IR without weights.
 * @param model A pointer to the newly created model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_read_model(const ov_core_t *core,
                                                        const char *model_path,
                                                        const char *bin_path,
                                                        ov_model_t **model);

/**
 * @brief Reads models from IR/ONNX/PDPD formats.
 * @param core A pointer to the ie_core_t instance.
 * @param model_str String with a model in IR/ONNX/PDPD format.
 * @param weights Shared pointer to a constant tensor with weights.
 * @param model A pointer to the newly created model.
 * Reading ONNX/PDPD models does not support loading weights from the @p weights tensors.
 * @note Created model object shares the weights with the @p weights object.
 * Thus, do not create @p weights on temporary data that can be freed later, since the model
 * constant data will point to an invalid memory.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_read_model_from_memory(const ov_core_t *core,
                                                        const char *model_str,
                                                        const ov_tensor_t *weights,
                                                        ov_model_t **model);

/**
 * @brief Release the memory allocated by ov_model_t.
 * @param model A pointer to the ov_model_t to free memory.
 */
OPENVINO_C_API(void) ov_model_free(ov_model_t *model);

/**
 * @brief Release the memory allocated by ov_compiled_model_t.
 * @param compiled_model A pointer to the ov_compiled_model_t to free memory.
 */
OPENVINO_C_API(void) ov_compiled_model_free(ov_compiled_model_t *compiled_model);

/**
 * @brief Creates a compiled model from a source model object.
 * Users can create as many compiled models as they need and use
 * them simultaneously (up to the limitation of the hardware resources).
 * @param core A pointer to the ie_core_t instance.
 * @param model Model object acquired from Core::read_model.
 * @param device_name Name of a device to load a model to.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param property Optional pack of pairs: (property name, property value) relevant only for this load operation
 * operation.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_compile_model(const ov_core_t* core,
                                            const ov_model_t* model,
                                            const char* device_name,
                                            ov_compiled_model_t **compiled_model,
                                            const ov_property_t* property);

/**
 * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
 * This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow,
 * especially for cases when caching is enabled and a cached model is available.
 * @param core A pointer to the ie_core_t instance.
 * @param model_path Path to a model.
 * @param device_name Name of a device to load a model to.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param property Optional pack of pairs: (property name, property value) relevant only for this load operation
 * operation.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_compile_model_from_file(const ov_core_t* core,
                                                    const char* model_path,
                                                    const char* device_name,
                                                    ov_compiled_model_t **compiled_model,
                                                    const ov_property_t* property);
                                                    
/**
 * @brief Sets properties for a device, acceptable keys can be found in ov_property_key_e.
 * @param core A pointer to the ie_core_t instance.
 * @param device_name Name of a device.
 * @param property ov_property propertys.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_set_property(const ov_core_t* core,
                                        const char* device_name,
                                        const ov_property_t* property);

/**
 * @brief Gets properties related to device behaviour.
 * The method extracts information that can be set via the set_property method.
 * @param core A pointer to the ie_core_t instance.
 * @param device_name  Name of a device to get a property value.
 * @param property_name  Property name.
 * @param property_value A pointer to property value.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_get_property(const ov_core_t* core, const char* device_name,
                                        const ov_property_key_e property_name,
                                        ov_property_value* property_value);

/**
 * @brief Returns devices available for inference.
 * @param core A pointer to the ie_core_t instance.
 * @param devices A pointer to the ov_available_devices_t instance.
 * Core objects go over all registered plugins and ask about available devices.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_get_available_devices(const ov_core_t* core, ov_available_devices_t* devices);

/**
 * @brief Releases memory occpuied by ov_available_devices_t
 * @param devices A pointer to the ov_available_devices_t instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(void) ov_available_devices_free(ov_available_devices_t* devices);

/**
 * @brief Imports a compiled model from the previously exported one.
 * @param core A pointer to the ov_core_t instance.
 * @param content A pointer to content of the exported model.
 * @param content_size Number of bytes in the exported network.
 * @param device_name Name of a device to import a compiled model for.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_import_model(
                                        const ov_core_t* core,
                                        const char *content,
                                        const size_t content_size,
                                        const char* device_name,
                                        ov_compiled_model_t **compiled_model);

/**
 * @brief Returns device plugins version information.
 * Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
 * in this case, std::map contains multiple entries, each per device.
 * @param core A pointer to the ov_core_t instance.
 * @param device_name Device name to identify a plugin.
 * @param versions A pointer to versions corresponding to device_name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_get_versions(
                                        const ov_core_t* core,
                                        const char* device_name,
                                        ov_core_version_list_t *versions);

/**
 * @brief Releases memory occupied by ov_core_version_list_t.
 * @param vers A pointer to the ie_core_versions to free memory.
 */
OPENVINO_C_API(void) ov_core_versions_free(ov_core_version_list_t *versions);

/**
 * @brief Get the outputs of ov_model_t.
 * @param model A pointer to the ov_model_t.
 * @param output_nodes A pointer to the ov_output_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_outputs(const ov_model_t* model, ov_output_node_list_t *output_nodes);

/**
 * @brief Get the outputs of ov_model_t.
 * @param model A pointer to the ov_model_t.
 * @param input_nodes A pointer to the ov_input_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_inputs(const ov_model_t* model, ov_output_node_list_t *input_nodes);

/**
 * @brief Get the tensor name of ov_output_node.
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_tensor_name(ov_output_node_list_t *nodes, size_t idx, char** tensor_name);

/**
 * @brief Get the tensor shape of ov_output_node.
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_tensor_shape(ov_output_node_list_t *nodes, size_t idx, ov_shape_t* tensor_shape);

/**
 * @brief Get the tensor type of ov_output_node.
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_type tensor type.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_tensor_type(ov_output_node_list_t *nodes, size_t idx, ov_element_type_e *tensor_type);

/**
 * @brief Get the outputs of ov_model_t.
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param input_node A pointer to the ov_output_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_input_by_name(const ov_model_t* model,
                                                    const char* tensor_name,
                                                    ov_output_node_t **input_node);

/**
 * @brief Get the outputs of ov_model_t.
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param input_node A pointer to the ov_input_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_input_by_id(const ov_model_t* model,
                                                    const size_t index,
                                                    ov_output_node_t **input_node);

/**
 * @brief Returns true if any of the op's defined in the model contains partial shape.
 * @param model A pointer to the ov_model_t. 
 */
OPENVINO_C_API(bool) ov_model_is_dynamic(const ov_model_t* model);

/**
 * @brief Do reshape in model with partial shape.
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param partialShape A PartialShape.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape(const ov_model_t* model, const char* tensor_name, const ov_partial_shape_t* partial_shape);

/**
 * @brief Gets the friendly name for a model. 
 * @param model A pointer to the ov_model_t.
 * @param friendly_name the model's friendly name.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name);

/**
 * @brief free ov_output_node_list_t
 * @param output_nodes The pointer to the instance of the ov_output_node_list_t to free.
 */
OPENVINO_C_API(void) ov_output_node_list_free(ov_output_node_list_t *output_nodes);

/**
 * @brief free ov_output_node_t
 * @param output_node The pointer to the instance of the ov_output_node_t to free.
 */
OPENVINO_C_API(void) ov_output_node_free(ov_output_node_t *output_node);

/**
 * @brief free char
 * @param content The pointer to the char to free.
 */
OPENVINO_C_API(void) ov_free(const char *content);

/**
 * @brief Create a ov_preprocess_t instance. 
 * @param model A pointer to the ov_model_t.
 * @param preprocess A pointer to the ov_preprocess_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_create(const ov_model_t* model,
                                            ov_preprocess_t **preprocess);

/**
 * @brief Release the memory allocated by ov_preprocess_t.
 * @param preprocess A pointer to the ov_preprocess_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_free(ov_preprocess_t *preprocess);

/**
 * @brief Get the input info of ov_preprocess_t instance. 
 * @param preprocess A pointer to the ov_preprocess_t.
 * @param tensor_name The name of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_get_input_info(const ov_preprocess_t* preprocess,
                                                ov_preprocess_input_info_t **preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_t instance by tensor name. 
 * @param preprocess A pointer to the ov_preprocess_t.
 * @param tensor_name The name of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_get_input_info_by_name(const ov_preprocess_t* preprocess,
                                                const char* tensor_name,
                                                ov_preprocess_input_info_t **preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_t instance by tensor order. 
 * @param preprocess A pointer to the ov_preprocess_t.
 * @param tensor_index The order of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_get_input_info_by_index(const ov_preprocess_t* preprocess,
                                                const size_t tensor_index,
                                                ov_preprocess_input_info_t **preprocess_input_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_info_t.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_input_info_free(ov_preprocess_input_info_t *preprocess_input_info);

/**
 * @brief Get a ov_preprocess_input_tensor_info_t. 
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_tensor_info A pointer to ov_preprocess_input_tensor_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_input_get_tensor_info(const ov_preprocess_input_info_t* preprocess_input_info,
                                                        ov_preprocess_input_tensor_info_t **preprocess_input_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_tensor_info_t.
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_input_tensor_info_free(ov_preprocess_input_tensor_info_t *preprocess_input_tensor_info);

/**
 * @brief Get a ov_preprocess_input_process_steps_t. 
 * @param ov_preprocess_input_info_t A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_steps A pointer to ov_preprocess_input_process_steps_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_input_get_preprocess_steps(const ov_preprocess_input_info_t* preprocess_input_info,
                                                        ov_preprocess_input_process_steps_t **preprocess_input_steps);

/**
 * @brief Release the memory allocated by ov_preprocess_input_process_steps_t.
 * @param preprocess_input_steps A pointer to the ov_preprocess_input_process_steps_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_input_process_steps_free(ov_preprocess_input_process_steps_t *preprocess_input_process_steps);

/**
 * @brief Add resize operation to model's dimensions. 
 * @param preprocess_input_process_steps A pointer to ov_preprocess_input_process_steps_t.
 * @param resize_algorithm A ov_preprocess_resizeAlgorithm instance
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_input_resize(ov_preprocess_input_process_steps_t* preprocess_input_process_steps,
                                                        const ov_preprocess_resize_algorithm_e resize_algorithm);

/**
 * @brief Set ov_preprocess_input_tensor_info_t precesion. 
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param element_type A point to element_type
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_input_tensor_info_set_element_type(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                        const ov_element_type_e element_type);

/**
 * @brief Helper function to reuse element type and shape from user's created tensor. 
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_input_tensor_info_set_tensor(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                        const ov_tensor_t* tensor);

/**
 * @brief Set ov_preprocess_input_tensor_info_t layout. 
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param layout A point to ov_layout_t
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_input_tensor_info_set_layout(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                        const ov_layout_t layout);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance. 
 * @param preprocess A pointer to the ov_preprocess_t.
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_get_output_info(const ov_preprocess_t* preprocess,
                                                ov_preprocess_output_info_t **preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance. 
 * @param preprocess A pointer to the ov_preprocess_t.
 * @param tensor_index The tensor index
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_get_output_info_by_index(const ov_preprocess_t* preprocess,
                                                const size_t tensor_index,
                                                ov_preprocess_output_info_t **preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance. 
 * @param preprocess A pointer to the ov_preprocess_t.
 * @param tensor_name The name of input.
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_get_output_info_by_name(const ov_preprocess_t* preprocess,
                                                const char* tensor_name,
                                                ov_preprocess_output_info_t **preprocess_output_info);

/**
 * @brief Release the memory allocated by ov_preprocess_output_info_t.
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_output_info_free(ov_preprocess_output_info_t *preprocess_output_info);

/**
 * @brief Get a ov_preprocess_input_tensor_info_t. 
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_output_get_tensor_info(ov_preprocess_output_info_t* preprocess_output_info,
                                                        ov_preprocess_output_tensor_info_t **preprocess_output_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_output_tensor_info_t.
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_output_tensor_info_free(ov_preprocess_output_tensor_info_t *preprocess_output_tensor_info);

/**
 * @brief Set ov_preprocess_input_tensor_info_t precesion. 
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t.
 * @param element_type A point to element_type
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_output_set_element_type(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info,
                                                        const ov_element_type_e element_type);

/**
 * @brief Get current input model information.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t
 * @return Status code of the operation: OK(0) for success.
*/
OPENVINO_C_API(ov_status_e) ov_preprocess_input_get_model_info(ov_preprocess_input_info_t* preprocess_input_info,
                                                        ov_preprocess_input_model_info_t **preprocess_input_model_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_model_info_t.
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_input_model_info_free(ov_preprocess_input_model_info_t *preprocess_input_model_info);

/**
 * @brief Set layout for model's input tensor. 
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t
 * @param layout A point to ov_layout_t
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_input_model_set_layout(ov_preprocess_input_model_info_t* preprocess_input_model_info,
                                                        const ov_layout_t layout);

/**
 * @brief Adds pre/post-processing operations to function passed in constructor. 
 * @param preprocess A pointer to the ov_preprocess_t.
 * @param model A pointer to the ov_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_preprocess_build(const ov_preprocess_t* preprocess,
                                            ov_model_t **model);

/**
 * @brief Gets runtime model information from a device.
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param model A pointer to the ov_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model,
                                                        ov_model_t **model);

/**
 * @brief Gets all inputs of a compiled model.
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param input_nodes A pointer to the ov_input_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_compiled_model_get_inputs(const ov_compiled_model_t* compiled_model,
                                                        ov_output_node_list_t *input_nodes);

/**
 * @brief Get all outputs of a compiled model.
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param output_nodes A pointer to the ov_output_node_list_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_compiled_model_get_outputs(const ov_compiled_model_t* compiled_model,
                                                        ov_output_node_list_t *output_nodes);

/**
 * @brief Creates an inference request object used to infer the compiled model.
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model,
                                                        ov_infer_request_t **infer_request);

/**
 * @brief Release the memory allocated by ov_infer_request_t.
 * @param infer_request A pointer to the ov_infer_request_t to free memory.
 */
OPENVINO_C_API(void) ov_infer_request_free(ov_infer_request_t *infer_request);

/**
 * @brief Sets properties for the current compiled model.
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param property ov_property_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model,
                                                        const ov_property_t* property);

/**
 * @brief Gets properties for current compiled model.
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param property_name Property name.
 * @param property_value A pointer to property value.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                                const ov_property_key_e property_name,
                                ov_property_value* property_value);

/**
 * @brief Exports the current compiled model to an output stream `std::ostream`.
 * The exported model can also be imported via the ov::Core::import_model method.
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param export_model_path Path to the file.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_compiled_model_export(const ov_compiled_model_t* compiled_model,
                                const char* export_model_path);
/**
 * @brief Sets an input/output tensor to infer on.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_set_tensor(ov_infer_request_t* infer_request,
                                const char* tensor_name, const ov_tensor_t* tensor);

/**
 * @brief Sets an input tensor to infer on.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the input tensor. If @p idx is greater than the number of model inputs, an exception is thrown.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request,
                                size_t idx, const ov_tensor_t* tensor);

/**
 * @brief Gets an input/output tensor to infer on.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_get_tensor(const ov_infer_request_t* infer_request,
                                const char* tensor_name, ov_tensor_t **tensor);

/**
 * @brief Gets an output tensor to infer on.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the tensor to get.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_get_out_tensor(const ov_infer_request_t* infer_request,
                                size_t idx, ov_tensor_t **tensor);

/**
 * @brief Infers specified input(s) in synchronous mode.
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_infer(ov_infer_request_t* infer_request);

/**
 * @brief Cancels inference request.
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_cancel(ov_infer_request_t* infer_request);

/**
 * @brief Starts inference of specified input(s) in asynchronous mode.
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_start_async(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the result
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_wait(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the result
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param callback  A function to be called.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_set_callback(ov_infer_request_t* infer_request,
                                                    const ov_call_back_t* callback);

/**
 * @brief Queries performance measures per layer to identify the most time consuming operation.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param profiling_infos  Vector of profiling information for operations in a model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_get_profiling_info(ov_infer_request_t* infer_request,
                                                    ov_profiling_info_list_t* profiling_infos);

/**
 * @brief Release the memory allocated by ov_profiling_info_list_t.
 * @param profiling_infos A pointer to the ov_profiling_info_list_t to free memory.
 */
OPENVINO_C_API(void) ov_profiling_info_list_free(ov_profiling_info_list_t *profiling_infos);

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param host_ptr Pointer to pre-allocated host memory
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_create_from_host_ptr(const ov_element_type_e type,
                                                    const ov_shape_t shape,
                                                    void* host_ptr, ov_tensor_t **tensor);


/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_create(const ov_element_type_e type,
                                            const ov_shape_t shape,
                                            ov_tensor_t **tensor);

/**
 * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(ov_status_e) ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape);

/**
 * @brief Get shape for tensor.
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape);

/**
 * @brief Get type for tensor.
 * @param type Tensor element type
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type);

/**
 * @brief the total number of elements (a product of all the dims or 1 for scalar).
 * @param elements_size number of elements
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size);

/**
 * @brief the size of the current Tensor in bytes.
 * @param byte_size the size of the current Tensor in bytes.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size);

/**
 * @brief Provides an access to the underlaying host memory.
 * @param data A point to host memory.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_data(const ov_tensor_t* tensor, void **data);

/**
 * @brief Free ov_tensor_t.
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(void) ov_tensor_free(ov_tensor_t* tensor);

#endif  // OV_C_API_H