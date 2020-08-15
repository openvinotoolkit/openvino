//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
            }

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
