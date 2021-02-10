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

#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief  Iterate a body over tensors, accumulating into tensors.
            class NGRAPH_API TensorIterator : public op::util::SubGraphOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"TensorIterator", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                bool visit_attributes(AttributeVisitor& visitor) override;

                TensorIterator() = default;
                explicit TensorIterator(const OutputVector& values);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                /// \return the body of the iteration
                std::shared_ptr<Function> get_body() const { return m_body; }
                /// \param body set the body of the iteration
                void set_body(const std::shared_ptr<Function>& body) { m_body = body; }
                void validate_and_infer_types() override;
                void revalidate_and_infer_types_for_body_ops();
                /// \return the body of the iteration
                std::shared_ptr<Function> get_function() override;

                int64_t get_num_iterations() const { return m_num_iterations; }
                void set_num_iterations(int64_t num_iterations)
                {
                    m_num_iterations = num_iterations;
                }

            private:
                int64_t m_num_iterations = -1;
            };
        }
        using v0::TensorIterator;
    }
}
