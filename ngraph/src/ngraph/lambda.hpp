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

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"

namespace ngraph
{
    class NGRAPH_API Lambda
    {
    public:
        static constexpr DiscreteTypeInfo type_info{"Lamdba", 0};
        const DiscreteTypeInfo& get_type_info() const { return type_info; }
        /// Return the function parameters
        const ParameterVector& get_parameters() const { return m_parameters; };
        /// Index for parameter, or -1
        int64_t get_parameter_index(const std::shared_ptr<op::Parameter>& parameter) const;
        /// Return a list of function's outputs
        const ResultVector& get_results() const { return m_results; };
        /// Index for value or result referencing it, or -1
        int64_t get_result_index(const Output<Node>& value) const;
        /// \brief Evaluate the lambda on inputs, putting results in outputs.
        /// \param outputs Tensors for the outputs to compute. One for each result
        /// \param inputs Tensors for the inputs. One for each inputs.
        bool evaluate(const HostTensorVector& output_tensors,
                      const HostTensorVector& input_tensors);

    protected:
        Lambda(const ResultVector& results, const ParameterVector& parameters);
        Lambda(const OutputVector& results, const ParameterVector& parameters);

        ResultVector m_results;
        ParameterVector m_parameters;
    };
}
