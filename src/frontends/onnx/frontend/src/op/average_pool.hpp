// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
/// \brief Convert ONNX AveragePool operation to an OV node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX AveragePool
///        operation.
OutputVector average_pool(const Node& node);

}  // namespace set_1

namespace set_7 {
/// \brief Convert ONNX AveragePool operation to an nGraph node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing Ngraph nodes producing output of ONNX AveragePool
///        operation.
OutputVector average_pool(const Node& node);

} // namespace set_7

namespace set_10 {
/// \brief Convert ONNX AveragePool operation to an nGraph node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing Ngraph nodes producing output of ONNX AveragePool
///        operation.
OutputVector average_pool(const Node& node);

}  // namespace set_10

namespace set_11 {
/// \brief Convert ONNX AveragePool operation to an nGraph node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing Ngraph nodes producing output of ONNX AveragePool
///        operation.
OutputVector average_pool(const Node& node);

}  // namespace set_11

namespace set_19 {
/// \brief Convert ONNX AveragePool operation to an nGraph node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing Ngraph nodes producing output of ONNX AveragePool
///        operation.
OutputVector average_pool(const Node& node);

}  // namespace set_19


}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
