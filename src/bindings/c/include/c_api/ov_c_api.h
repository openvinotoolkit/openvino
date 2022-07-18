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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#ifdef __cplusplus
#    define OPENVINO_C_API_EXTERN extern "C"
#else
#    define OPENVINO_C_API_EXTERN
#endif

#if defined(OPENVINO_STATIC_LIBRARY) || defined(__GNUC__) && (__GNUC__ < 4)
#    define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __VA_ARGS__
#    define OV_NODISCARD
#else
#    if defined(_WIN32)
#        define OPENVINO_C_API_CALLBACK __cdecl
#        ifdef openvino_c_EXPORTS
#            define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __declspec(dllexport) __VA_ARGS__ __cdecl
#        else
#            define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __declspec(dllimport) __VA_ARGS__ __cdecl
#        endif
#        define OV_NODISCARD
#    else
#        define OPENVINO_C_API(...) OPENVINO_C_API_EXTERN __attribute__((visibility("default"))) __VA_ARGS__
#        define OV_NODISCARD        __attribute__((warn_unused_result))
#    endif
#endif

#ifndef OPENVINO_C_API_CALLBACK
#    define OPENVINO_C_API_CALLBACK
#endif

#define MAX_DIMENSION 8

typedef struct ov_core ov_core_t;
typedef struct ov_output_const_node ov_output_const_node_t;
typedef struct ov_output_node ov_output_node_t;
typedef struct ov_model ov_model_t;
typedef struct ov_compiled_model ov_compiled_model_t;
typedef struct ov_infer_request ov_infer_request_t;
typedef struct ov_tensor ov_tensor_t;
typedef struct ov_rank ov_rank_t;
typedef struct ov_dimensions ov_dimensions_t;
typedef struct ov_property ov_property_t;
typedef struct ov_layout ov_layout_t;
typedef struct ov_partial_shape ov_partial_shape_t;

typedef struct ov_preprocess_prepostprocessor ov_preprocess_prepostprocessor_t;
typedef struct ov_preprocess_inputinfo ov_preprocess_inputinfo_t;
typedef struct ov_preprocess_inputtensorinfo ov_preprocess_inputtensorinfo_t;
typedef struct ov_preprocess_outputinfo ov_preprocess_outputinfo_t;
typedef struct ov_preprocess_outputtensorinfo ov_preprocess_outputtensorinfo_t;
typedef struct ov_preprocess_inputmodelinfo ov_preprocess_inputmodelinfo_t;
typedef struct ov_preprocess_preprocesssteps ov_preprocess_preprocesssteps_t;

typedef void* ov_property_value_t;

/**
 * @struct ov_callback_t
 * @brief Completion callback definition about the function and args
 */
typedef struct {
    void(OPENVINO_C_API_CALLBACK* callback_func)(void* args);
    void* args;
} ov_callback_t;

/**
 * @struct ov_ProfilingInfo_t
 * @brief Store profiling info data
 */
typedef struct {
    enum Status {       //!< Defines the general status of a node.
        NOT_RUN,        //!< A node is not executed.
        OPTIMIZED_OUT,  //!< A node is optimized out during graph optimization phase.
        EXECUTED        //!< A node is executed.
    } status;
    int64_t real_time;      //!< The absolute time, in microseconds, that the node ran (in total).
    int64_t cpu_time;       //!< The net host CPU time that the node ran.
    const char* node_name;  //!< Name of a node.
    const char* exec_type;  //!< Execution type of a unit.
    const char* node_type;  //!< Node type.
} ov_profiling_info_t;

/**
 * @struct ov_profiling_info_list_t
 * @brief A list of profiling info data
 */
typedef struct {
    ov_profiling_info_t* profiling_infos;
    size_t num;
} ov_profiling_info_list_t;

/**
 * @struct ov_version
 * @brief Represents OpenVINO version information
 */
typedef struct ov_version {
    const char* buildNumber;  //!< A string representing OpenVINO version
    const char* description;
} ov_version_t;

/**
 * @struct ov_core_version
 * @brief  Represents version information that describes device and ov runtime library
 */
typedef struct {
    const char* device_name;  //!< A device name
    const char* buildNumber;  //!< A build number
    const char* description;  //!< A device description
} ov_core_version_t;

/**
 * @struct ov_core_version_list
 * @brief  Represents version information that describes all devices and ov runtime library
 */
typedef struct {
    ov_core_version_t* versions;  //!< An array of device versions
    size_t num_vers;              //!< A number of versions in the array
} ov_core_version_list_t;

/**
 * @struct ov_available_devices_t
 * @brief Represent all available devices.
 */
typedef struct {
    char** devices;
    size_t num_devices;
} ov_available_devices_t;

/**
 * @struct ov_output_node_list_t
 * @brief Reprents an array of ov_output_nodes.
 */
typedef struct {
    ov_output_const_node_t* output_nodes;
    size_t num;
} ov_output_node_list_t;

/**
 * @struct ov_shape_t
 * @brief Reprents a static shape.
 */
typedef struct {
    int64_t rank;
    int64_t dims[MAX_DIMENSION];
} ov_shape_t;

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
    CALLOC_FAILED = 7,
    INVALID_PARAM = 8,
    UNEXPECTED = 9,
    REQUEST_BUSY = 10,
    RESULT_NOT_READY = 11,
    NOT_ALLOCATED = 12,
    INFER_NOT_STARTED = 13,
    NETWORK_NOT_READ = 14,
    INFER_CANCELLED = 15,
    UNKNOWN_ERROR = 16,
} ov_status_e;

/**
 * @enum ov_preprocess_resizealgorithm_e
 * @brief This enum contains codes for all preprocess resize algorithm.
 */
typedef enum { RESIZE_LINEAR, RESIZE_CUBIC, RESIZE_NEAREST } ov_preprocess_resizealgorithm_e;

/**
 * @enum ov_element_type_e
 * @brief This enum contains codes for element type.
 */
typedef enum {
    UNDEFINED = 0,  //!< Undefined element type
    DYNAMIC,        //!< Dynamic element type
    BOOLEAN,        //!< boolean element type
    BF16,           //!< bf16 element type
    F16,            //!< f16 element type
    F32,            //!< f32 element type
    F64,            //!< f64 element type
    I4,             //!< i4 element type
    I8,             //!< i8 element type
    I16,            //!< i16 element type
    I32,            //!< i32 element type
    I64,            //!< i64 element type
    U1,             //!< binary element type
    U4,             //!< u4 element type
    U8,             //!< u8 element type
    U16,            //!< u16 element type
    U32,            //!< u32 element type
    U64,            //!< u64 element type
    MAX,
} ov_element_type_e;

/**
 * @enum ov_color_format_e
 * @brief This enum contains enumerations for color format.
 */
typedef enum {
    UNDEFINE = 0,       //!< Undefine color format
    NV12_SINGLE_PLANE,  // Image in NV12 format as single tensor
    NV12_TWO_PLANES,    // Image in NV12 format represented as separate tensors for Y and UV planes.
    I420_SINGLE_PLANE,  // Image in I420 (YUV) format as single tensor
    I420_THREE_PLANES,  // Image in I420 format represented as separate tensors for Y, U and V planes.
    RGB,
    BGR,
    RGBX,  // Image in RGBX interleaved format (4 channels)
    BGRX   // Image in BGRX interleaved format (4 channels)
} ov_color_format_e;

/**
 * @enum ov_performance_mode_e
 * @brief Enum to define possible performance mode hints
 * @brief This represents OpenVINO 2.0 ov::hint::PerformanceMode entity.
 *
 */
typedef enum {
    UNDEFINED_MODE = -1,        //!<  Undefined value, performance setting may vary from device to device
    LATENCY = 1,                //!<  Optimize for latency
    THROUGHPUT = 2,             //!<  Optimize for throughput
    CUMULATIVE_THROUGHPUT = 3,  //!< Optimize for cumulative throughput
} ov_performance_mode_e;

/**
 * @enum ov_affinity_e
 * @brief Enum to define possible affinity patterns
 */
typedef enum {
    NONE = -1,  //!<  Disable threads affinity pinning
    CORE = 0,   //!<  Pin threads to cores, best for static benchmarks
    NUMA = 1,   //!<  Pin threads to NUMA nodes, best for real-life, contented cases. On the Windows and MacOS* this
                //!<  option behaves as CORE
    HYBRID_AWARE = 2,  //!< Let the runtime to do pinning to the cores types, e.g. prefer the "big" cores for latency
                       //!< tasks. On the hybrid CPUs this option is default
} ov_affinity_e;

/**
 * @struct ov_property_key_e
 * @brief Represent all available property key.
 */
typedef enum {
    SUPPORTED_PROPERTIES = 0,  //!<  Read-only property<char *> to get a string list of supported read-only properties.
    AVAILABLE_DEVICES,         //!<  Read-only property<char *> to get a list of available device IDs
    OPTIMAL_NUMBER_OF_INFER_REQUESTS,  //!<  Read-only property<uint32_t> to get an unsigned integer value of optimaln
                                       //!<  umber of compiled model infer requests.
    RANGE_FOR_ASYNC_INFER_REQUESTS,    //!<  Read-only property<unsigned int, unsigned int, unsigned int> to provide a
                                       //!<  hint for a range for number of async infer requests. If device supports
                                       //!<  streams, the metric provides range for number of IRs per stream.
    RANGE_FOR_STREAMS,  //!<  Read-only property<unsigned int, unsigned int> to provide information about a range for
                        //!<  streams on platforms where streams are supported
    FULL_DEVICE_NAME,   //!<  Read-only property<char *> to get a string value representing a full device name.
    OPTIMIZATION_CAPABILITIES,  //!<  Read-only property<char *> to get a string list of capabilities options per
                                //!<  device.
    CACHE_DIR,    //!<  Read-write property<char *> to set/get the directory which will be used to store any data cached
                  //!<  by plugins.
    NUM_STREAMS,  //!<  Read-write property<uint32_t> to set/get the number of executor logical partitions
    AFFINITY,  //!<  Read-write property<ov_affinity_e> to set/get the name for setting CPU affinity per thread option.
    INFERENCE_NUM_THREADS,  //!<  Read-write property<int32_t> to set/get the maximum number of threads that can be used
                            //!<  for inference tasks.
    PERFORMANCE_HINT,       //!< Read-write property<ov_performance_mode_e>, it is high-level OpenVINO Performance Hints
                       //!< unlike low-level properties that are individual (per-device), the hints are something that
                       //!< every device accepts and turns into device-specific settings detail see
                       //!< ov_performance_mode_e to get its hint's key name
    NETWORK_NAME,              //!<  Read-only property<char *> to get a name of name of a model
    INFERENCE_PRECISION_HINT,  //!< Read-write property<ov_element_type_e> to set the hint for device to use specified
                               //!< precision for inference
    OPTIMAL_BATCH_SIZE,  //!<  Read-only property<uint32_t> to query information optimal batch size for the given device
                         //!<  and the network
    MAX_BATCH_SIZE,  //!<  Read-only property to get maximum batch size which does not cause performance degradation due
                     //!<  to memory swap impact.
    PERFORMANCE_HINT_NUM_REQUESTS,  //!<  (Optional) property<uint32_t> that backs the Performance Hints by giving
                                    //!<  additional information on how many inference requests the application will be
                                    //!<  keeping in flight usually this value comes from the actual use-case  (e.g.
                                    //!<  number of video-cameras, or other sources of inputs)
    MAX_KEY_VALUE,
} ov_property_key_e;

/**
 * @struct ov_property_data_t
 * @brief Represent a pair of <key, value> data.
 */
typedef struct {
    ov_property_key_e key;
    ov_property_value_t value;
} ov_property_data_t;

/**
 * @brief Get version of OpenVINO.
 * @param ov_version_t a pointer to the version
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_get_openvino_version(ov_version_t* version);

/**
 * @brief Release the memory allocated by ov_version_t.
 * @param version A pointer to the ov_version_t to free memory.
 */
OPENVINO_C_API(void) ov_version_free(ov_version_t* version);

// OV Core

/**
 * @defgroup Core Core
 * @ingroup ov_c_api
 * Set of functions dedicated to working with registered plugins and loading
 * model to the registered devices.
 * @{
 */

/**
 * @brief Constructs OpenVINO Core instance using XML configuration file with devices description.
 * See RegisterPlugins for more details.
 * @ingroup Core
 * @param xml_config_file A path to .xml file with devices to load from. If XML configuration file is not specified,
 * then default plugin.xml file will be used.
 * @param core A pointer to the newly created ov_core_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_create(const char* xml_config_file, ov_core_t** core);

/**
 * @brief Release the memory allocated by ov_core_t.
 * @ingroup Core
 * @param core A pointer to the ov_core_t to free memory.
 */
OPENVINO_C_API(void) ov_core_free(ov_core_t* core);

/**
 * @brief Reads models from IR/ONNX/PDPD formats.
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param model_path Path to a model.
 * @param bin_path Path to a data file.
 * For IR format (*.bin):
 *  * if path is empty, will try to read a bin file with the same name as xml and
 *  * if the bin file with the same name is not found, will load IR without weights.
 * For ONNX format (*.onnx):
 *  * the bin_path parameter is not used.
 * For PDPD format (*.pdmodel)
 *  * the bin_path parameter is not used.
 * @param model A pointer to the newly created model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_read_model(const ov_core_t* core, const char* model_path, const char* bin_path, ov_model_t** model);

/**
 * @brief Reads models from IR/ONNX/PDPD formats.
 * @ingroup Core
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
OPENVINO_C_API(ov_status_e)
ov_core_read_model_from_memory(const ov_core_t* core,
                               const char* model_str,
                               const ov_tensor_t* weights,
                               ov_model_t** model);

/**
 * @brief Creates a compiled model from a source model object.
 * Users can create as many compiled models as they need and use
 * them simultaneously (up to the limitation of the hardware resources).
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param model Model object acquired from Core::read_model.
 * @param device_name Name of a device to load a model to.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param property Optional pack of pairs: (property name, property value) relevant only for this load operation
 * operation.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model(const ov_core_t* core,
                      const ov_model_t* model,
                      const char* device_name,
                      ov_compiled_model_t** compiled_model,
                      const ov_property_t* property);

/**
 * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
 * This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow,
 * especially for cases when caching is enabled and a cached model is available.
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param model_path Path to a model.
 * @param device_name Name of a device to load a model to.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param property Optional pack of pairs: (property name, property value) relevant only for this load operation
 * operation.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model_from_file(const ov_core_t* core,
                                const char* model_path,
                                const char* device_name,
                                ov_compiled_model_t** compiled_model,
                                const ov_property_t* property);

/**
 * @brief Sets properties for a device, acceptable keys can be found in ov_property_key_e.
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param device_name Name of a device.
 * @param property ov_property propertys.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_set_property(const ov_core_t* core, const char* device_name, const ov_property_t* property);

/**
 * @brief Gets properties related to device behaviour.
 * The method extracts information that can be set via the set_property method.
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param device_name  Name of a device to get a property value.
 * @param property_name  Property name.
 * @param property_value A pointer to property value.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_get_property(const ov_core_t* core,
                     const char* device_name,
                     const ov_property_key_e property_name,
                     ov_property_value_t* property_value);

/**
 * @brief Returns devices available for inference.
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param devices A pointer to the ov_available_devices_t instance.
 * Core objects go over all registered plugins and ask about available devices.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_get_available_devices(const ov_core_t* core, ov_available_devices_t* devices);

/**
 * @brief Releases memory occpuied by ov_available_devices_t
 * @ingroup Core
 * @param devices A pointer to the ov_available_devices_t instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(void) ov_available_devices_free(ov_available_devices_t* devices);

/**
 * @brief Imports a compiled model from the previously exported one.
 * @ingroup Core
 * @param core A pointer to the ov_core_t instance.
 * @param content A pointer to content of the exported model.
 * @param content_size Number of bytes in the exported network.
 * @param device_name Name of a device to import a compiled model for.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_import_model(const ov_core_t* core,
                     const char* content,
                     const size_t content_size,
                     const char* device_name,
                     ov_compiled_model_t** compiled_model);

/**
 * @brief Returns device plugins version information.
 * Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
 * in this case, std::map contains multiple entries, each per device.
 * @ingroup Core
 * @param core A pointer to the ov_core_t instance.
 * @param device_name Device name to identify a plugin.
 * @param versions A pointer to versions corresponding to device_name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_get_versions_by_device_name(const ov_core_t* core, const char* device_name, ov_core_version_list_t* versions);

/**
 * @brief Releases memory occupied by ov_core_version_list_t.
 * @ingroup Core
 * @param vers A pointer to the ie_core_versions to free memory.
 */
OPENVINO_C_API(void) ov_core_versions_free(ov_core_version_list_t* versions);

/** @} */  // end of Core

// Model
/**
 * @defgroup Model Model
 * @ingroup ov_c_api
 * Set of functions representing of Model and Node.
 * @{
 */

/**
 * @brief Release the memory allocated by ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t to free memory.
 */
OPENVINO_C_API(void) ov_model_free(ov_model_t* model);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param output_nodes A pointer to the ov_output_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_outputs(const ov_model_t* model, ov_output_node_list_t* output_nodes);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param input_nodes A pointer to the ov_input_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_inputs(const ov_model_t* model, ov_output_node_list_t* input_nodes);

/**
 * @brief Get the tensor name of ov_output_node list by index.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_any_name_by_index(ov_output_node_list_t* nodes, size_t idx, char** tensor_name);

/**
 * @brief Get the tensor name of node.
 * @ingroup Model
 * @param node A pointer to the ov_output_const_node_t.
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_any_name(ov_output_const_node_t* node, char** tensor_name);

/**
 * @brief Get the shape of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_get_shape_by_index(ov_output_node_list_t* nodes, size_t idx, ov_shape_t* shape);

/**
 * @brief Get the partial shape of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_get_partial_shape_by_index(ov_output_node_list_t* nodes, size_t idx, ov_partial_shape_t** partial_shape);

/**
 * @brief Get the tensor shape of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_element_type(ov_output_const_node_t* node, ov_element_type_e* tensor_type);
/**
 * @brief Get the tensor type of ov_output_node.
 * @ingroup Model
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_type tensor type.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_get_element_type_by_index(ov_output_node_list_t* nodes, size_t idx, ov_element_type_e* tensor_type);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param input_node A pointer to the ov_output_const_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_const_node_t** input_node);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param input_node A pointer to the ov_input_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_const_node_t** input_node);

/**
 * @brief Returns true if any of the op's defined in the model contains partial shape.
 * @param model A pointer to the ov_model_t.
 */
OPENVINO_C_API(bool) ov_model_is_dynamic(const ov_model_t* model);

/**
 * @brief Do reshape in model with partial shape for a specified name.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param partialShape A PartialShape.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_name(const ov_model_t* model, const char* tensor_name, const ov_partial_shape_t* partial_shape);

/**
 * @brief Do reshape in model with a list of <name, partial shape>.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param tensor_names input tensor name (char *) list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_names(const ov_model_t* model,
                          const char* tensor_names[],
                          const ov_partial_shape_t* partial_shapes[],
                          size_t cnt);

/**
 * @brief Do reshape in model with a list of <port id, partial shape>.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param ports The port list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_ports(const ov_model_t* model, size_t* ports, const ov_partial_shape_t** partial_shape, size_t cnt);

/**
 * @brief Do reshape in model for port 0.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param partialShape A PartialShape.
 */
OPENVINO_C_API(ov_status_e) ov_model_reshape(const ov_model_t* model, const ov_partial_shape_t* partial_shape);

/**
 * @brief Do reshape in model with a list of <ov_output_node_t, partial shape>.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param output_nodes The ov_output_node_t list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_nodes(const ov_model_t* model,
                          const ov_output_node_t* output_nodes[],
                          const ov_partial_shape_t* partial_shapes[],
                          size_t cnt);

/**
 * @brief Gets the friendly name for a model.
 * @ingroup Model
 * @param model A pointer to the ov_model_t.
 * @param friendly_name the model's friendly name.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name);

/**
 * @brief free ov_output_node_list_t
 * @param output_nodes The pointer to the instance of the ov_output_node_list_t to free.
 */
OPENVINO_C_API(void) ov_output_node_list_free(ov_output_node_list_t* output_nodes);

/**
 * @brief free ov_output_const_node_t
 * @ingroup Model
 * @param output_node The pointer to the instance of the ov_output_const_node_t to free.
 */
OPENVINO_C_API(void) ov_output_node_free(ov_output_const_node_t* output_node);

/** @} */  // end of Model

// Compiled Model
/**
 * @defgroup compiled_model compiled_model
 * @ingroup ov_c_api
 * Set of functions representing of Compiled Model.
 * @{
 */

/**
 * @brief Gets runtime model information from a device.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param model A pointer to the ov_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model, ov_model_t** model);

/**
 * @brief Gets all inputs of a compiled model.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param input_nodes A pointer to the ov_input_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_inputs(const ov_compiled_model_t* compiled_model, ov_output_node_list_t* input_nodes);

/**
 * @brief Get all outputs of a compiled model.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param output_nodes A pointer to the ov_output_node_list_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_outputs(const ov_compiled_model_t* compiled_model, ov_output_node_list_t* output_nodes);

/**
 * @brief Creates an inference request object used to infer the compiled model.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param infer_request A pointer to the ov_infer_request_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model, ov_infer_request_t** infer_request);

/**
 * @brief Release the memory allocated by ov_infer_request_t.
 * @ingroup compiled_model
 * @param infer_request A pointer to the ov_infer_request_t to free memory.
 */
OPENVINO_C_API(void) ov_infer_request_free(ov_infer_request_t* infer_request);

/**
 * @brief Sets properties for the current compiled model.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param property ov_property_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model, const ov_property_t* property);

/**
 * @brief Gets properties for current compiled model.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param property_name Property name.
 * @param property_value A pointer to property value.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                               const ov_property_key_e key,
                               ov_property_value_t* value);

/**
 * @brief Exports the current compiled model to an output stream `std::ostream`.
 * The exported model can also be imported via the ov::Core::import_model method.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t.
 * @param export_model_path Path to the file.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_compiled_model_export_model(const ov_compiled_model_t* compiled_model, const char* export_model_path);

/**
 * @brief Release the memory allocated by ov_compiled_model_t.
 * @ingroup compiled_model
 * @param compiled_model A pointer to the ov_compiled_model_t to free memory.
 */
OPENVINO_C_API(void) ov_compiled_model_free(ov_compiled_model_t* compiled_model);

/** @} */  // end of compiled_model

// prepostprocess
/**
 * @defgroup prepostprocess prepostprocess
 * @ingroup ov_c_api
 * Set of functions representing of PrePostProcess.
 * @{
 */

/**
 * @brief Create a ov_preprocess_prepostprocessor_t instance.
 * @ingroup prepostprocess
 * @param model A pointer to the ov_model_t.
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_create(const ov_model_t* model, ov_preprocess_prepostprocessor_t** preprocess);

/**
 * @brief Release the memory allocated by ov_preprocess_prepostprocessor_t.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_prepostprocessor_free(ov_preprocess_prepostprocessor_t* preprocess);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_name The name of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_inputinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_input(const ov_preprocess_prepostprocessor_t* preprocess,
                                     ov_preprocess_inputinfo_t** preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance by tensor name.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_name The name of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_inputinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_input_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                             const char* tensor_name,
                                             ov_preprocess_inputinfo_t** preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance by tensor order.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_index The order of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_inputinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_input_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                              const size_t tensor_index,
                                              ov_preprocess_inputinfo_t** preprocess_input_info);

/**
 * @brief Release the memory allocated by ov_preprocess_inputinfo_t.
 * @ingroup prepostprocess
 * @param preprocess_input_info A pointer to the ov_preprocess_inputinfo_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_inputinfo_free(ov_preprocess_inputinfo_t* preprocess_input_info);

/**
 * @brief Get a ov_preprocess_inputtensorinfo_t.
 * @ingroup prepostprocess
 * @param preprocess_input_info A pointer to the ov_preprocess_inputinfo_t.
 * @param preprocess_input_tensor_info A pointer to ov_preprocess_inputtensorinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputinfo_tensor(const ov_preprocess_inputinfo_t* preprocess_input_info,
                               ov_preprocess_inputtensorinfo_t** preprocess_input_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_inputtensorinfo_t.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_inputtensorinfo_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_inputtensorinfo_free(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info);

/**
 * @brief Get a ov_preprocess_preprocesssteps_t.
 * @ingroup prepostprocess
 * @param ov_preprocess_inputinfo_t A pointer to the ov_preprocess_inputinfo_t.
 * @param preprocess_input_steps A pointer to ov_preprocess_preprocesssteps_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputinfo_preprocess(const ov_preprocess_inputinfo_t* preprocess_input_info,
                                   ov_preprocess_preprocesssteps_t** preprocess_input_steps);

/**
 * @brief Release the memory allocated by ov_preprocess_preprocesssteps_t.
 * @ingroup prepostprocess
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocesssteps_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_preprocesssteps_free(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps);

/**
 * @brief Add resize operation to model's dimensions.
 * @ingroup prepostprocess
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocesssteps_t.
 * @param resize_algorithm A ov_preprocess_resizeAlgorithm instance
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocesssteps_resize(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
                                     const ov_preprocess_resizealgorithm_e resize_algorithm);

/**
 * @brief Set ov_preprocess_inputtensorinfo_t precesion.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_inputtensorinfo_t.
 * @param element_type A point to element_type
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputtensorinfo_set_element_type(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                               const ov_element_type_e element_type);

/**
 * @brief Set ov_preprocess_inputtensorinfo_t color format.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_inputtensorinfo_t.
 * @param colorFormat The enumerate of colorFormat
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputtensorinfo_set_color_format(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                               const ov_color_format_e colorFormat);

/**
 * @brief Set ov_preprocess_inputtensorinfo_t spatial_static_shape.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_inputtensorinfo_t.
 * @param input_height The height of input
 * @param input_width The width of input
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputtensorinfo_set_spatial_static_shape(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                                       const size_t input_height,
                                                       const size_t input_width);

/**
 * @brief Convert ov_preprocess_preprocesssteps_t element type.
 * @ingroup prepostprocess
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocesssteps_t.
 * @param element_type preprocess input element type.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocesssteps_convert_element_type(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
                                                   const ov_element_type_e element_type);

/**
 * @brief Convert ov_preprocess_preprocesssteps_t color.
 * @ingroup prepostprocess
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocesssteps_t.
 * @param colorFormat The enumerate of colorFormat.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocesssteps_convert_color(ov_preprocess_preprocesssteps_t* preprocess_input_process_steps,
                                            const ov_color_format_e colorFormat);

/**
 * @brief Helper function to reuse element type and shape from user's created tensor.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_inputtensorinfo_t.
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputtensorinfo_set_from(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                       const ov_tensor_t* tensor);

/**
 * @brief Set ov_preprocess_inputtensorinfo_t layout.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_inputtensorinfo_t.
 * @param layout A point to ov_layout_t
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputtensorinfo_set_layout(ov_preprocess_inputtensorinfo_t* preprocess_input_tensor_info,
                                         ov_layout_t* layout);

/**
 * @brief Get the output info of ov_preprocess_outputinfo_t instance.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param preprocess_output_info A pointer to the ov_preprocess_outputinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_output(const ov_preprocess_prepostprocessor_t* preprocess,
                                      ov_preprocess_outputinfo_t** preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_outputinfo_t instance.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_index The tensor index
 * @param preprocess_output_info A pointer to the ov_preprocess_outputinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_output_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                               const size_t tensor_index,
                                               ov_preprocess_outputinfo_t** preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_outputinfo_t instance.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_name The name of input.
 * @param preprocess_output_info A pointer to the ov_preprocess_outputinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_output_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                              const char* tensor_name,
                                              ov_preprocess_outputinfo_t** preprocess_output_info);

/**
 * @brief Release the memory allocated by ov_preprocess_outputinfo_t.
 * @ingroup prepostprocess
 * @param preprocess_output_info A pointer to the ov_preprocess_outputinfo_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_outputinfo_free(ov_preprocess_outputinfo_t* preprocess_output_info);

/**
 * @brief Get a ov_preprocess_inputtensorinfo_t.
 * @ingroup prepostprocess
 * @param preprocess_output_info A pointer to the ov_preprocess_outputinfo_t.
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_outputtensorinfo_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_outputinfo_tensor(ov_preprocess_outputinfo_t* preprocess_output_info,
                                ov_preprocess_outputtensorinfo_t** preprocess_output_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_outputtensorinfo_t.
 * @ingroup prepostprocess
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_outputtensorinfo_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_outputtensorinfo_free(ov_preprocess_outputtensorinfo_t* preprocess_output_tensor_info);

/**
 * @brief Set ov_preprocess_inputtensorinfo_t precesion.
 * @ingroup prepostprocess
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_outputtensorinfo_t.
 * @param element_type A point to element_type
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_output_set_element_type(ov_preprocess_outputtensorinfo_t* preprocess_output_tensor_info,
                                      const ov_element_type_e element_type);

/**
 * @brief Get current input model information.
 * @ingroup prepostprocess
 * @param preprocess_input_info A pointer to the ov_preprocess_inputinfo_t.
 * @param preprocess_input_model_info A pointer to the ov_preprocess_inputmodelinfo_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputinfo_model(ov_preprocess_inputinfo_t* preprocess_input_info,
                              ov_preprocess_inputmodelinfo_t** preprocess_input_model_info);

/**
 * @brief Release the memory allocated by ov_preprocess_inputmodelinfo_t.
 * @ingroup prepostprocess
 * @param preprocess_input_model_info A pointer to the ov_preprocess_inputmodelinfo_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_inputmodelinfo_free(ov_preprocess_inputmodelinfo_t* preprocess_input_model_info);

/**
 * @brief Set layout for model's input tensor.
 * @ingroup prepostprocess
 * @param preprocess_input_model_info A pointer to the ov_preprocess_inputmodelinfo_t
 * @param layout A point to ov_layout_t
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_inputmodelinfo_set_layout(ov_preprocess_inputmodelinfo_t* preprocess_input_model_info,
                                        ov_layout_t* layout);

/**
 * @brief Adds pre/post-processing operations to function passed in constructor.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param model A pointer to the ov_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_build(const ov_preprocess_prepostprocessor_t* preprocess, ov_model_t** model);

/** @} */  // end of compiled_model

// infer_request
/**
 * @defgroup infer_request infer_request
 * @ingroup ov_c_api
 * Set of functions representing of infer_request.
 * @{
 */

/**
 * @brief Sets an input/output tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_tensor(ov_infer_request_t* infer_request, const char* tensor_name, const ov_tensor_t* tensor);

/**
 * @brief Sets an input tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the input tensor. If @p idx is greater than the number of model inputs, an exception is thrown.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, size_t idx, const ov_tensor_t* tensor);

/**
 * @brief Gets an input/output tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param tensor_name  Name of the input or output tensor.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_tensor(const ov_infer_request_t* infer_request, const char* tensor_name, ov_tensor_t** tensor);

/**
 * @brief Gets an output tensor to infer on.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param idx Index of the tensor to get.
 * @param tensor Reference to the tensor.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_out_tensor(const ov_infer_request_t* infer_request, size_t idx, ov_tensor_t** tensor);

/**
 * @brief Infers specified input(s) in synchronous mode.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_infer(ov_infer_request_t* infer_request);

/**
 * @brief Cancels inference request.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_cancel(ov_infer_request_t* infer_request);

/**
 * @brief Starts inference of specified input(s) in asynchronous mode.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_start_async(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the result
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 */
OPENVINO_C_API(ov_status_e) ov_infer_request_wait(ov_infer_request_t* infer_request);

/**
 * @brief Waits for the result to become available. Blocks until the result
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param callback  A function to be called.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_set_callback(ov_infer_request_t* infer_request, const ov_callback_t* callback);

/**
 * @brief Queries performance measures per layer to identify the most time consuming operation.
 * @ingroup infer_request
 * @param infer_request A pointer to the ov_infer_request_t.
 * @param profiling_infos  Vector of profiling information for operations in a model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_infer_request_get_profiling_info(ov_infer_request_t* infer_request, ov_profiling_info_list_t* profiling_infos);

/**
 * @brief Release the memory allocated by ov_profiling_info_list_t.
 * @ingroup infer_request
 * @param profiling_infos A pointer to the ov_profiling_info_list_t to free memory.
 */
OPENVINO_C_API(void) ov_profiling_info_list_free(ov_profiling_info_list_t* profiling_infos);

/** @} */  // end of infer_request

// Tensor
/**
 * @defgroup Tensor Tensor
 * @ingroup ov_c_api
 * Set of functions representing of Tensor, Shape, PartialShape, etc.
 * @{
 */

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @ingroup Tensor
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param host_ptr Pointer to pre-allocated host memory
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_tensor_create_from_host_ptr(const ov_element_type_e type,
                               const ov_shape_t shape,
                               void* host_ptr,
                               ov_tensor_t** tensor);

/**
 * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
 * @ingroup Tensor
 * @param type Tensor element type
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t** tensor);

/**
 * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
 * @ingroup Tensor
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(ov_status_e) ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape);

/**
 * @brief Get shape for tensor.
 * @ingroup Tensor
 * @param shape Tensor shape
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape);

/**
 * @brief Get type for tensor.
 * @ingroup Tensor
 * @param type Tensor element type
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type);

/**
 * @brief the total number of elements (a product of all the dims or 1 for scalar).
 * @ingroup Tensor
 * @param elements_size number of elements
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size);

/**
 * @brief the size of the current Tensor in bytes.
 * @ingroup Tensor
 * @param byte_size the size of the current Tensor in bytes.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size);

/**
 * @brief Provides an access to the underlaying host memory.
 * @ingroup Tensor
 * @param data A point to host memory.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_tensor_data(const ov_tensor_t* tensor, void** data);

/**
 * @brief Free ov_tensor_t.
 * @ingroup Tensor
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(void) ov_tensor_free(ov_tensor_t* tensor);

/**
 * @brief Create a rank object
 * @ingroup Tensor
 * @param min_dimension The lower inclusive limit for the dimension
 * @param max_dimension The upper inclusive limit for the dimension
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_rank_create(ov_rank_t** rank, int64_t min_dimension, int64_t max_dimension);

/**
 * @brief Release rank object.
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_rank_free(ov_rank_t* rank);

/**
 * @brief Create a dimensions object
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_dimensions_create(ov_dimensions_t** dimensions);

/**
 * @brief Release a dimensions object
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_dimensions_free(ov_dimensions_t* dimensions);

/**
 * @brief Add a dimension with bounded range into dimensions
 * @ingroup Tensor
 * @param min_dimension The lower inclusive limit for the dimension
 * @param max_dimension The upper inclusive limit for the dimension
 *
 * Static dimension: min_dimension == max_dimension > 0
 * Dynamic dimension:
 *     min_dimension == -1 ? 0 : min_dimension
 *     max_dimension == -1 ? Interval::s_max : max_dimension
 *
 */
OPENVINO_C_API(ov_status_e) ov_dimensions_add(ov_dimensions_t* dimension, int64_t min_dimension, int64_t max_dimension);

/**
 * @brief Create a partial shape and initialze with rank and dimension.
 * @ingroup Tensor
 * @param rank support dynamic and static rank
 * @param dims support dynamic and static dimension
 *  Dynamic rank:
 *     Example: "?"
 *  Static rank, but dynamic dimensions on some or all axes.
 *     Examples: "{1,2,?,4}" or "{?,?,?}" or "{1,2,-1,4}""
 *  Static rank, and static dimensions on all axes.
 *     Examples: "{1,2,3,4}" or "{6}" or "{}""
 *
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e)
ov_partial_shape_create(ov_partial_shape_t** partial_shape_obj, ov_rank_t* rank, ov_dimensions_t* dims);

/**
 * @brief Parse the partial shape to readable string.
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(const char*) ov_partial_shape_to_string(ov_partial_shape_t* partial_shape);

/**
 * @brief Release partial shape.
 * @ingroup Tensor
 * @param partial_shape will be released.
 */
OPENVINO_C_API(void) ov_partial_shape_free(ov_partial_shape_t* partial_shape);

/**
 * @brief Covert partial shape to static shape.
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_partial_shape_to_shape(ov_partial_shape_t* partial_shape, ov_shape_t* shape);

/**
 * @brief Covert shape to partial shape.
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_shape_to_partial_shape(ov_shape_t* shape, ov_partial_shape_t** partial_shape);

/**
 * @brief Create a layout object.
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_layout_create(ov_layout_t** layout, const char* layout_desc);

/**
 * @brief Free layout object.
 * @param layout will be released.
 */
OPENVINO_C_API(void) ov_layout_free(ov_layout_t* layout);

/**
 * @brief Convert layout object to a readable string.
 * @param layout will be converted.
 * @return string that describes the layout content.
 */
OPENVINO_C_API(const char*) ov_layout_to_string(ov_layout_t* layout);

/** @} */  // end of Tensor

// Property
/**
 * @defgroup Property Property
 * @ingroup ov_c_api
 * Set of functions representing of Property.
 * @{
 */

/**
 * @brief Create a property object.
 * @ingroup Property
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_property_create(ov_property_t** property);

/**
 * @brief Free property object.
 * @ingroup Property
 * @param property will be released.
 */
OPENVINO_C_API(void) ov_property_free(ov_property_t* property);

/**
 * @brief Free property data.
 * @ingroup Property
 * @param property data will be released.
 */
OPENVINO_C_API(void) ov_property_value_free(ov_property_value_t value);

/**
 * @brief Put <key, value> into property object.
 * @ingroup Property
 * @param property will be add new <key, value>.
 */
OPENVINO_C_API(ov_status_e) ov_property_put(ov_property_t* property, ov_property_key_e key, ov_property_value_t value);

/** @} */  // end of Property

/**
 * @brief Print the error info.
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(const char*) ov_get_error_info(ov_status_e status);

/**
 * @brief free char
 * @param content The pointer to the char to free.
 */
OPENVINO_C_API(void) ov_free(const char* content);

#endif  // OV_C_API_H