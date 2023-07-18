// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
/// \brief      Creates nGraph node representing ONNX GlobalLpPool operator.
///
/// \note       This functions calculates "entrywise" norms in spatial/feature
///             dimensions. That is it treats matrix/tensor in spatial/feature
///             dimensions as a vector and applies apropriate norm on it. The
///             result is a scalar.
///
///             Suppose A contains spatial dimensions of input tensor, then
///             for matrix A we have p-norm defined as following double sum over
///             all elements:
///             ||A||_p = ||vec(A)||_p =
///                 [sum_{i=1}^m sum_{j=1}^n abs(a_{i,j})^p]^{1/p}
///
/// \param[in]  node  The input ONNX node representing this operation.
///
/// \return     Vector of nodes containting resulting nGraph nodes.
///
OutputVector global_lp_pool(const Node& node);
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
