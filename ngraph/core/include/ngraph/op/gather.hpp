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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public Op
            {
            public:
                static const int64_t AXIS_NOT_SET_VALUE = std::numeric_limits<int64_t>::max();
                static constexpr NodeTypeInfo type_info{"Gather", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Gather() = default;
                /// \param params The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather
                /// \param axis The tensor is a dimension index to gather data from
                Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axis);

                bool visit_attributes(AttributeVisitor& visitor) override;
                int64_t get_axis() const;

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

                bool constant_fold(OutputVector& output_values,
                                   const OutputVector& inputs_values) override;

            private:
                static const int PARAMS;
                static const int INDICES;
                static const int AXIS;

                bool evaluate_gather(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const;
            };
        } // namespace v1
    }     // namespace op
} // namespace ngraph
