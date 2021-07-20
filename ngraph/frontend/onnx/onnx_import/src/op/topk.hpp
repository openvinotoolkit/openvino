// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                /// \brief Performs ONNX TopK operation.
                ///
                /// \param node The ONNX node object representing this operation.
                /// \return The vector containing Ngraph nodes producing output of ONNX TopK
                ///         operation (both values and indices).
                OutputVector topk(const Node& node);
            } // namespace set_1

            /// \brief Performs TopK operation from ONNX version 1.5
            ///
            /// \details ONNX op set 10 added support for K as a dynamic input, not a static
            /// attribute.
            namespace set_10
            {
                OutputVector topk(const Node& node);
            }

            /// \brief Performs TopK operation from ONNX version 1.6
            ///
            /// \details ONNX op set 11 added support for `largest` and `sorted` attributes.
            namespace set_11
            {
                OutputVector topk(const Node& node);
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
