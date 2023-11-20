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
element::TypeVector unsupported_types();

/// \brief is_convert returns true if the first parameter is a Convert or ConvertLike node and false otherwise
/// TODO: find better place for this helper function
OPENVINO_API
bool is_convert(const std::shared_ptr<Node>& node);

/// \brief If the node has constant inputs with types that (before constant_fold) should be converted to f32 - the
///        function converts and constantfolds those inputs to f32. Then, the function clones the node with
///        the new inputs and returns the new node.
///
/// \param node
///
/// \return New node with f32 inputs if the inputs require conversion or the input node otherwise
OPENVINO_API std::shared_ptr<Node> try_clone_and_convert_inputs(const std::shared_ptr<Node>& node);

}  // namespace util
}  // namespace ov
