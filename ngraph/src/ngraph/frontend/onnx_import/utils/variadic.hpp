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

#include <numeric>

#include "core/node.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace variadic
        {
            /// \brief Create an nGraph version of an ONNX variadic operation.
            ///        This creates a subgraph with a series of binary operations.
            ///
            /// \param node Incoming ONNX opearation.
            ///
            /// \tparam T   Class of an nGraph binary operation (e.g. Add, Minimum, Maximum)
            ///
            /// \return nGraph node equivalent of the ONNX operation
            template <class T>
            inline OutputVector
                make_ng_variadic_op(const Node& node,
                                    const ngraph::op::AutoBroadcastSpec& auto_broadcast =
                                        ngraph::op::AutoBroadcastSpec::NUMPY)
            {
                const OutputVector ng_inputs{node.get_ng_inputs()};

                // Templated binary operation - Creates Add, Minimum, Maximum, etc.
                const auto binary_operation = [&auto_broadcast](const Output<ngraph::Node>& arg0,
                                                                const Output<ngraph::Node>& arg1) {
                    return std::make_shared<T>(arg0, arg1, auto_broadcast);
                };

                // Create a result node as a series of binary operations
                const auto result = std::accumulate(
                    std::next(std::begin(ng_inputs)), // First operand value - the second input
                    std::end(ng_inputs),              // Last value - final input
                    ng_inputs.front(),                // Initial value - first input
                    binary_operation);

                return {result};
            }

        } // namespace variadic

    } // namespace  onnx_import

} // namespace  ngraph
