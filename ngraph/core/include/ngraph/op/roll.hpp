//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v7
        {
            /// \brief Tensor roll operation.
            class NGRAPH_API Roll : public Op
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Roll() = default;

                ///
                /// \brief      Constructs a roll operation.
                ///
                /// \param      data         Node producing the tensor to be shifted.
                /// \param      shift        Node producing the 0D or 1D tensor which specifies the
                /// number of places by which the elements are shifted.
                /// \param      axes         Node producing the 0D or 1D tensor which specifies axes
                /// along which elements are shifted.
                ///
                Roll(const Output<Node>& data, const Output<Node>& shift, const Output<Node>& axes);

                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        } // namespace v7
    }     // namespace op
} // namespace ngraph
