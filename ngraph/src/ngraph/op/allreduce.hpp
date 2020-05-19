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
#include "ngraph/distributed.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API AllReduce : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"AllReduce", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                AllReduce() = default;
                AllReduce(const Output<Node>& arg,
                          reduction::Type reduce_type = reduction::Type::SUM);

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                reduction::Type get_reduce_type() const;
                void set_reduce_type(reduction::Type reduce_type);
                bool visit_attributes(AttributeVisitor& visitor) override;

            private:
                reduction::Type m_reduce_type{reduction::Type::SUM};
            };
        }
        using v0::AllReduce;
    }
}
