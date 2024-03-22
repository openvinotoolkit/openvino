// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace util {

/// \brief Returns a vector with unsupported element types. Constant inputs with those types (in general) require
///        conversion before node can be constant folded
OPENVINO_API
const element::TypeVector& unsupported_types();

OPENVINO_API
bool is_type_unsupported(const element::Type& type);

OPENVINO_API
void save_original_input_precisions(const std::shared_ptr<Node>& node);

OPENVINO_API
bool has_original_input_precision(const Input<Node>& input);

OPENVINO_API
element::Type get_original_input_precision(const Input<Node>& input);

OPENVINO_API
void remove_original_input_precision_attribute(Input<Node>& input);

OPENVINO_API bool node_requires_precision_conversion(const Node* const node);

/// \brief If the node has constant inputs with types that (before constant_fold) should be converted to f32 - the
///        function converts and constantfolds those inputs to f32. Then, the function clones the node with
///        the new inputs and returns the new node.
///
/// \param node
///
/// \return New node with f32 inputs if the inputs require conversion or the input node otherwise
OPENVINO_API std::shared_ptr<Node> convert_to_supported_precision(Node* const node);

OPENVINO_API std::shared_ptr<Node> convert_to_supported_precision(Node* const node, const OutputVector& inputs);

OPENVINO_API bool evaluate_node_with_unsupported_precision(const Node* node,
                                                           TensorVector& outputs,
                                                           const TensorVector& inputs);

}  // namespace util
}  // namespace ov
