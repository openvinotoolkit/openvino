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

#include "ngraph/op/util/logical_reduction_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Performs a reduction using "logical or"
            ///
            /// The reduction is performed over slices of the first input. The slices shape depends
            /// on the values passed to the second input - the axes.
            class NGRAPH_API ReduceLogicalOr : public util::LogicalReductionKeepDims
            {
            public:
                NGRAPH_RTTI_DECLARATION;
                ReduceLogicalOr() = default;
                /// \brief Constructs a ReduceLogicalOr node.
                ///
                /// \param data - The input tensor with data to be reduced
                /// \param reduction_axes - The input tensor with information about axes over which
                /// the first tensor should be sliced prior to the reduction operation
                /// \param keep_dims - Indicates if the axes used for reduction should be held/kept
                ReduceLogicalOr(const Output<Node>& data,
                                const Output<Node>& reduction_axes,
                                const bool keep_dims = false);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
            };
        }
    }
}
