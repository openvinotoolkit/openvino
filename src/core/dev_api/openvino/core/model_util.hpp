// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <unordered_map>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/model.hpp"

namespace ov {

/** @brief Generic atg suggests automated functionality.
 *
 * Can be use for template specialization or function overloading
 */
struct AutoTag {};
inline constexpr AutoTag AUTO{};

/** @brief Alias to map of port number and tensor names */
using TensorNamesMap = std::unordered_map<size_t, TensorNames>;
}  // namespace ov

namespace ov::util {

/** @brief Set input tensors names for the model
 *
 * Sets only tensors defined in the input map.
 * The tensors defined in map but not existing in the model will be ignored.
 *
 * @param model Model to set its input tensors names.
 * @param inputs_names Map of input tensor names.
 */
OPENVINO_API void set_input_tensors_names(Model& model, const TensorNamesMap& inputs_names);

/** @brief Set input tensors names for the model
 *
 * Sets tensors defined in the input map.
 * Tensors not defined in the map are set to default names if the tensor hasn't got names.
 * The tensors defined in the map but not existing in the model will be ignored.
 *
 * @param model Model to set its input tensors names.
 * @param inputs_names Map of input tensor names. Default empty.
 */
OPENVINO_API void set_input_tensors_names(const AutoTag&, Model& model, const TensorNamesMap& inputs_names = {});

/** @brief Set output tensors names for the model
 *
 * Sets only tensors defined in the output map.
 * The tensors defined in map but not existing in the model will be ignored.
 *
 * @param model Model to set its output tensors names.
 * @param outputs_names Map of output tensor names.AnyMap
 */
OPENVINO_API void set_output_tensor_names(Model& model, const TensorNamesMap& outputs_names);

/** @brief Set output tensors names for the model.
 *
 * Sets tensors defined in the output map.
 * Tensors not defined in the map are set to default names if the tensor hasn't got names.
 * The tensors defined in the map but not existing in the model will be ignored.
 *
 * @param model Model to set its output tensors names.
 * @param outputs_names Map of output tensor names. Default empty.
 */
OPENVINO_API void set_output_tensor_names(const AutoTag&, Model& model, const TensorNamesMap& outputs_names = {});

/** @brief Set input and output tensors names for the model
 *
 * Sets only tensors defined in the input and output maps.
 * The tensors defined in maps but not existing in the model will be ignored.
 *
 * @param model Model to set its input and output tensors names.
 * @param inputs_names Map of input tensor names.
 * @param outputs_names Map of output tensor names.
 */
OPENVINO_API void set_tensors_names(Model& model,
                                    const TensorNamesMap& inputs_names,
                                    const TensorNamesMap& outputs_names);

/** @brief Set input and output tensors names for the model.
 *
 * Sets tensors defined in the input and output maps.
 * Tensors not defined in the maps are set to default names if the tensor hasn't got names.
 * The tensors defined in the maps but not existing in the model will be ignored.
 *
 * @param model Model to set its input and output tensors names.
 * @param inputs_names Map of input tensor names. Default empty.
 * @param outputs_names Map of output tensor names. Default empty.
 */
OPENVINO_API void set_tensors_names(const AutoTag&,
                                    Model& model,
                                    const TensorNamesMap& inputs_names = {},
                                    const TensorNamesMap& outputs_names = {});

/**
 * @brief Get the model cache id attribute.
 *
 * If returned model path is empty, then model path attribute is not set or it should be ignored.
 *
 * @param model  Model to get its cache id attributes.
 * @return pair of model path and model cache ID modifier.
 */
OPENVINO_API std::pair<std::filesystem::path, std::string_view> get_model_cache_id_attr(const Model& model);

/**
 * @brief Check if attribute name is model cache attribute.
 *
 * @param name  Attribute name to check.
 * @return True if name is model cache attribute, false otherwise.
 */
OPENVINO_API bool is_model_cache_attr(const std::string& name);

}  // namespace ov::util
