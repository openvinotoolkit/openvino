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
                /// \brief  Allows to define the purpose of inputs/outputs in the body
                struct SpecialBodyPorts
                {
                    SpecialBodyPorts() = default;
                    SpecialBodyPorts(int64_t in_current_iteration_input_idx,
                                     int64_t in_body_condition_output_idx)
                        : current_iteration_input_idx(in_current_iteration_input_idx)
                        , body_condition_output_idx(in_body_condition_output_idx)
                    {
                    }
                    // -1 means the input is not provided, this input is optional
                    int64_t current_iteration_input_idx = -1;
                    // -1 means the output is not provided,
                    // this output is required, throw an exception if not provided
                    int64_t body_condition_output_idx = -1;
                };

                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a Loop operation.
                Loop() = default;

                /// \brief Constructs a Loop operation.
                ///
                /// \param trip_count Node specifies the maximum number of iterations.
                /// \param execution_condition Node determines whether to execute the first
                /// iteration or not.
                Loop(const Output<Node>& trip_count, const Output<Node>& execution_condition);

                int64_t get_num_iterations() const { return m_num_iterations; }
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

                Output<Node> get_concatenated_slices(const Output<Node>& value,
                                                     int64_t start,
                                                     int64_t stride,
                                                     int64_t part_size,
                                                     int64_t end,
                                                     int64_t axis) override;

                void set_special_body_ports(const SpecialBodyPorts& special_body_ports)
                {
                    m_special_body_ports = special_body_ports;
                }

                SpecialBodyPorts get_special_body_ports() const { return m_special_body_ports; }
                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            private:
                SpecialBodyPorts m_special_body_ports;
                int64_t m_num_iterations = -1; // -1 means infinity
            };
        }
    }
}
