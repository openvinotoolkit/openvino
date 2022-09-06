// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ov_core C API, which is a C wrapper for ov::Core class.
 * This class represents an OpenVINO runtime Core entity.
 * @file ov_core.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_compiled_model.h"
#include "openvino/c/ov_model.h"
#include "openvino/c/ov_node.h"
#include "openvino/c/ov_property.h"
#include "openvino/c/ov_tensor.h"

typedef struct ov_core ov_core_t;

/**
 * @struct ov_version
 * @brief Represents OpenVINO version information
 */
typedef struct ov_version {
    const char* buildNumber;  //!< A string representing OpenVINO version
    const char* description;  //!< A string representing OpenVINO description
} ov_version_t;

/**
 * @struct ov_core_version
 * @brief  Represents version information that describes device and ov runtime library
 */
typedef struct {
    const char* device_name;  //!< A device name
    ov_version_t version;     //!< Version
} ov_core_version_t;

/**
 * @struct ov_core_version_list
 * @brief  Represents version information that describes all devices and ov runtime library
 */
typedef struct {
    ov_core_version_t* versions;  //!< An array of device versions
    size_t size;                  //!< A number of versions in the array
} ov_core_version_list_t;

/**
 * @struct ov_available_devices_t
 * @brief Represent all available devices.
 */
typedef struct {
    char** devices;
    size_t size;
} ov_available_devices_t;

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
 * @ingroup openvino_c
 * Set of functions dedicated to working with registered plugins and loading
 * model to the registered devices.
 * @{
 */

/**
 * @brief Constructs OpenVINO Core instance by default.
 * See RegisterPlugins for more details.
 * @ingroup Core
 * @param core A pointer to the newly created ov_core_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_create(ov_core_t** core);

/**
 * @brief Constructs OpenVINO Core instance using XML configuration file with devices description.
 * See RegisterPlugins for more details.
 * @ingroup Core
 * @param xml_config_file A path to .xml file with devices to load from. If XML configuration file is not specified,
 * then default plugin.xml file will be used.
 * @param core A pointer to the newly created ov_core_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_core_create_with_config(const char* xml_config_file, ov_core_t** core);

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
 * @param property Optional pack of pairs: (property name, property value) relevant only for this load operation
 * operation.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model(const ov_core_t* core,
                      const ov_model_t* model,
                      const char* device_name,
                      const ov_properties_t* property,
                      ov_compiled_model_t** compiled_model);

/**
 * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
 * This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow,
 * especially for cases when caching is enabled and a cached model is available.
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param model_path Path to a model.
 * @param device_name Name of a device to load a model to.
 * @param property Optional pack of pairs: (property name, property value) relevant only for this load operation
 * operation.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model_from_file(const ov_core_t* core,
                                const char* model_path,
                                const char* device_name,
                                const ov_properties_t* property,
                                ov_compiled_model_t** compiled_model);

/**
 * @brief Sets properties for a device, acceptable keys can be found in ov_property_key_e.
 * @ingroup Core
 * @param core A pointer to the ie_core_t instance.
 * @param device_name Name of a device.
 * @param property ov_property propertys.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_set_property(const ov_core_t* core, const char* device_name, const ov_properties_t* property);

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
                     const char* property_name,
                     ov_any_t* property_value);

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
