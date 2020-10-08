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

#include <vector>

#include "ngraph/factory_adapter.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v5
        {
            /// \brief  Iterate a body over tensors, accumulating into tensors.
            class NGRAPH_API Loop : public op::util::SubGraphOp
            {
            public:
                struct SpecialBodyPorts
                {
                    int64_t current_iteration_input_idx =
                        -1; // -1 means input is not provided, this input is optional
                    int64_t body_condition_output_idx = 0; // default index, this output is required
                };

                static constexpr NodeTypeInfo type_info{"Loop", 5};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                bool visit_attributes(AttributeVisitor& visitor) override;

                Loop()
                {
                    // default trip_count, execution_condition
                    auto trip_count = std::make_shared<ngraph::op::Constant>(
                        ngraph::element::i64, ngraph::Shape{1}, -1);
                    auto exec_condition = std::make_shared<ngraph::op::Constant>(
                        ngraph::element::boolean, ngraph::Shape{1}, true);
                    set_argument(0, Output<Node>(trip_count));
                    set_argument(1, Output<Node>(exec_condition));
                };
                Loop(const Output<Node>& trip_count,
                     const Output<Node>& condition,
                     const OutputVector& values);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                /// \return the body of the iteration
                std::shared_ptr<Function> get_body() const { return m_body; }
                /// \param body set the body of the iteration
                void set_body(const std::shared_ptr<Function>& body) { m_body = body; }
                /// \return a reference to the input descriptions.

                void validate_and_infer_types() override;

                int64_t get_num_iterations() const { return m_num_iterations; }
                void set_num_iterations(int64_t num_iterations)
                {
                    m_num_iterations = num_iterations;
                }

                void set_trip_count_input(const Output<Node>& value) { set_argument(0, value); }
                void set_execution_condition_input(const Output<Node>& value)
                {
                    set_argument(1, value);
                }

                void set_sliced_input(const std::shared_ptr<Parameter>& parameter,
                                      const Output<Node>& value,
                                      int64_t start,
                                      int64_t stride,
                                      int64_t part_size,
                                      int64_t end,
                                      int64_t axis) override
                {
                    NGRAPH_CHECK(false,
                                 "Incorrect type of input. Implicit slicing is not supported in "
                                 "Loop operation.");
                }

                void set_special_body_ports(const SpecialBodyPorts& special_body_ports)
                {
                    m_special_body_ports = special_body_ports;
                }

                SpecialBodyPorts get_special_body_ports() const { return m_special_body_ports; }
            private:
                SpecialBodyPorts m_special_body_ports;
                int64_t m_num_iterations = -1;
            };
        }
        using v5::Loop;
    }
}
