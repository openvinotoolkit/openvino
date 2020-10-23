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

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API Squeeze : public ngraph::op::util::FusedOp
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                Squeeze() = default;
                Squeeze(const Output<Node>& data, const Output<Node>& axes);

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual OutputVector decompose_op() const override;
                virtual void pre_validate_and_infer_types() override;
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v0::Squeeze;
    }
}

NGRAPH_SUPPRESS_DEPRECATED_END
