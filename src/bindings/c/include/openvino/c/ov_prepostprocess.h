// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for prepostprocess C API, which is a C wrapper for ov::preprocess class.
 * Main class for adding pre- and post- processing steps to existing ov::Model
 * @file ov_prepostprocess.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_layout.h"
#include "openvino/c/ov_model.h"
#include "openvino/c/ov_tensor.h"

typedef struct ov_preprocess_prepostprocessor ov_preprocess_prepostprocessor_t;
typedef struct ov_preprocess_input_info ov_preprocess_input_info_t;
typedef struct ov_preprocess_input_tensor_info ov_preprocess_input_tensor_info_t;
typedef struct ov_preprocess_output_info ov_preprocess_output_info_t;
typedef struct ov_preprocess_output_tensor_info ov_preprocess_output_tensor_info_t;
typedef struct ov_preprocess_input_model_info ov_preprocess_input_model_info_t;
typedef struct ov_preprocess_preprocess_steps ov_preprocess_preprocess_steps_t;

/**
 * @enum ov_color_format_e
 * @brief This enum contains enumerations for color format.
 */
typedef enum {
    UNDEFINE = 0U,      //!< Undefine color format
    NV12_SINGLE_PLANE,  //!< Image in NV12 format as single tensor
    NV12_TWO_PLANES,    //!< Image in NV12 format represented as separate tensors for Y and UV planes.
    I420_SINGLE_PLANE,  //!< Image in I420 (YUV) format as single tensor
    I420_THREE_PLANES,  //!< Image in I420 format represented as separate tensors for Y, U and V planes.
    RGB,                //!< Image in RGB interleaved format (3 channels)
    BGR,                //!< Image in BGR interleaved format (3 channels)
    RGBX,               //!< Image in RGBX interleaved format (4 channels)
    BGRX                //!< Image in BGRX interleaved format (4 channels)
} ov_color_format_e;

/**
 * @enum ov_preprocess_resize_algorithm_e
 * @brief This enum contains codes for all preprocess resize algorithm.
 */
typedef enum {
    RESIZE_LINEAR,  //!< linear algorithm
    RESIZE_CUBIC,   //!< cubic algorithm
    RESIZE_NEAREST  //!< nearest algorithm
} ov_preprocess_resize_algorithm_e;

// prepostprocess
/**
 * @defgroup prepostprocess prepostprocess
 * @ingroup openvino_c
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
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_input_info(const ov_preprocess_prepostprocessor_t* preprocess,
                                              ov_preprocess_input_info_t** preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance by tensor name.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_name The name of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_input_info_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                      const char* tensor_name,
                                                      ov_preprocess_input_info_t** preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance by tensor order.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_index The order of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_input_info_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                       const size_t tensor_index,
                                                       ov_preprocess_input_info_t** preprocess_input_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_info_t.
 * @ingroup prepostprocess
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_input_info_free(ov_preprocess_input_info_t* preprocess_input_info);

/**
 * @brief Get a ov_preprocess_input_tensor_info_t.
 * @ingroup prepostprocess
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_tensor_info A pointer to ov_preprocess_input_tensor_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_info_get_tensor_info(const ov_preprocess_input_info_t* preprocess_input_info,
                                         ov_preprocess_input_tensor_info_t** preprocess_input_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_tensor_info_t.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_input_tensor_info_free(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info);

/**
 * @brief Get a ov_preprocess_preprocess_steps_t.
 * @ingroup prepostprocess
 * @param ov_preprocess_input_info_t A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_info_get_preprocess_steps(const ov_preprocess_input_info_t* preprocess_input_info,
                                              ov_preprocess_preprocess_steps_t** preprocess_input_steps);

/**
 * @brief Release the memory allocated by ov_preprocess_preprocess_steps_t.
 * @ingroup prepostprocess
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocess_steps_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_preprocess_steps_free(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps);

/**
 * @brief Add resize operation to model's dimensions.
 * @ingroup prepostprocess
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param resize_algorithm A ov_preprocess_resizeAlgorithm instance
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_resize(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                      const ov_preprocess_resize_algorithm_e resize_algorithm);

/**
 * @brief Set ov_preprocess_input_tensor_info_t precesion.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param element_type A point to element_type
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_element_type(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                 const ov_element_type_e element_type);

/**
 * @brief Set ov_preprocess_input_tensor_info_t color format.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param colorFormat The enumerate of colorFormat
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_color_format(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                 const ov_color_format_e colorFormat);

/**
 * @brief Set ov_preprocess_input_tensor_info_t spatial_static_shape.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param input_height The height of input
 * @param input_width The width of input
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_spatial_static_shape(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
    const size_t input_height,
    const size_t input_width);

/**
 * @brief Convert ov_preprocess_preprocess_steps_t element type.
 * @ingroup prepostprocess
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocess_steps_t.
 * @param element_type preprocess input element type.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_convert_element_type(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                    const ov_element_type_e element_type);

/**
 * @brief Convert ov_preprocess_preprocess_steps_t color.
 * @ingroup prepostprocess
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocess_steps_t.
 * @param colorFormat The enumerate of colorFormat.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_convert_color(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                             const ov_color_format_e colorFormat);

/**
 * @brief Helper function to reuse element type and shape from user's created tensor.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param tensor A point to ov_tensor_t
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_from(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                         const ov_tensor_t* tensor);

/**
 * @brief Set ov_preprocess_input_tensor_info_t layout.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param layout A point to ov_layout_t
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_layout(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                           ov_layout_t* layout);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_output_info(const ov_preprocess_prepostprocessor_t* preprocess,
                                               ov_preprocess_output_info_t** preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_index The tensor index
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_output_info_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                        const size_t tensor_index,
                                                        ov_preprocess_output_info_t** preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance.
 * @ingroup prepostprocess
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_name The name of input.
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_output_info_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                       const char* tensor_name,
                                                       ov_preprocess_output_info_t** preprocess_output_info);

/**
 * @brief Release the memory allocated by ov_preprocess_output_info_t.
 * @ingroup prepostprocess
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_output_info_free(ov_preprocess_output_info_t* preprocess_output_info);

/**
 * @brief Get a ov_preprocess_input_tensor_info_t.
 * @ingroup prepostprocess
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_output_info_get_tensor_info(ov_preprocess_output_info_t* preprocess_output_info,
                                          ov_preprocess_output_tensor_info_t** preprocess_output_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_output_tensor_info_t.
 * @ingroup prepostprocess
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_output_tensor_info_free(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info);

/**
 * @brief Set ov_preprocess_input_tensor_info_t precesion.
 * @ingroup prepostprocess
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t.
 * @param element_type A point to element_type
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_output_set_element_type(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info,
                                      const ov_element_type_e element_type);

/**
 * @brief Get current input model information.
 * @ingroup prepostprocess
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_info_get_model_info(ov_preprocess_input_info_t* preprocess_input_info,
                                        ov_preprocess_input_model_info_t** preprocess_input_model_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_model_info_t.
 * @ingroup prepostprocess
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t to free memory.
 */
OPENVINO_C_API(void) ov_preprocess_input_model_info_free(ov_preprocess_input_model_info_t* preprocess_input_model_info);

/**
 * @brief Set layout for model's input tensor.
 * @ingroup prepostprocess
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t
 * @param layout A point to ov_layout_t
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_model_info_set_layout(ov_preprocess_input_model_info_t* preprocess_input_model_info,
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

/** @} */  // end of prepostprocess