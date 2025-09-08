// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace symbol {
namespace util {

/// \brief Collects symbols from shape. Symbols of static dimensions are guaranteed to be nullptr
///
/// \param shape    Shape object to collect symbols from
/// \param symbols   TensorSymbol object to collect symbols to
///
/// \return Status of collecting the symbols (false if rank is static else true)
TRANSFORMATIONS_API bool get_symbols(const ov::PartialShape& shape, ov::TensorSymbol& symbols);

/// \brief Collects symbols from tensor of Output object
///
/// \param output   Output object to collect symbols from
/// \param symbols   TensorSymbol object to collect symbols to
///
/// \return Status of collecting the symbols (false if tensor has no symbols else true)
TRANSFORMATIONS_API bool get_symbols(const ov::Output<ov::Node>& output, ov::TensorSymbol& symbols);

/// \brief Compares
///
/// \param lhs   TensorSymbol object to compare
/// \param rhs   TensorSymbol object to compare
///
/// \return true if symbols are unique and equal between lhs and rhs else false
TRANSFORMATIONS_API bool are_unique_and_equal_symbols(const ov::TensorSymbol& lhs, const ov::TensorSymbol& rhs);

/// \brief Compares dimensions: if dimensions are static compares values of dimensions, if dimensions are dynamic
/// compares their respective symbols
///
/// \param lhs   Dimension object to compare
/// \param rhs   Dimension object to compare
///
/// \return true if static dimensions are equal and dynamic dimensions have equal symbols else false
TRANSFORMATIONS_API bool dims_are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs);

}  // namespace util
}  // namespace symbol
}  // namespace ov
