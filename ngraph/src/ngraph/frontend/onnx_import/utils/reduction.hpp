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

#include <cstdint> // std::int64_t
#include <memory>  // std::make_shared
#include <utility> // std::move

#include "core/node.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reduction
        {
            namespace detail
            {
                AxisSet get_reduction_axes(const Node& node);

            } // namespace  detail

            // An overload for reduction operators that take reduction axes as input
            using RuntimeReductionFunction = std::function<std::shared_ptr<ngraph::Node>(
                const Output<ngraph::Node>&, const std::shared_ptr<ngraph::Node>&, bool)>;

            // An overload for reduction operators that take reduction axes as an attribute
            using ReductionFunction = std::function<std::shared_ptr<ngraph::Node>(
                const Output<ngraph::Node>&, const ngraph::AxisSet&)>;

            ///
            /// \brief      Create an nGraph version of an ONNX reduction operation.
            ///
            /// \param[in]  node                The node representing incoming ONNX operation.
            /// \param[in]  ng_input            The input (nGraph) Tensor.
            /// \param[in]  reduction_function  The reduction function defining arithmetic reduction
            ///                                 operation (e.g. Min, Max, Sum, Product).
            ///
            /// \return     nGraph node equivalent of the ONNX operation.
            ///
            Output<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const Output<ngraph::Node>& ng_input,
                                     ReductionFunction reduction_function);

            ///
            /// \brief      Create an nGraph version of an ONNX reduction operation.
            ///
            /// \param[in]  node                The node representing incoming ONNX operation.
            /// \param[in]  ng_input            The input (nGraph) Tensor.
            /// \param[in]  reduction_function  The reduction function defining arithmetic dynamic
            ///                                 reduction operation (e.g. ReduceProd, ReduceSum).
            ///
            /// \return     nGraph node equivalent of the ONNX operation.
            ///
            Output<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const Output<ngraph::Node>& ng_input,
                                     RuntimeReductionFunction reduction_function);

        } // namespace  reduction
    }     // namespace onnx_import
} // namespace ngraph
