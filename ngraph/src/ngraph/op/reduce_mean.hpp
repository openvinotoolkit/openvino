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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            class NGRAPH_API ReduceMean : public util::ArithmeticReductionKeepDims
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReduceMean", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ReduceMean() = default;

                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to 1 it holds axes that are used for reduction.
                ReduceMean(const Output<Node>& arg,
                           const Output<Node>& reduction_axes,
                           bool keep_dims = false);

                size_t get_version() const override { return 1; }
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

#ifdef NGRAPH_EVALUATE_ENABLE
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;
#endif
            };
        }
    }
}
