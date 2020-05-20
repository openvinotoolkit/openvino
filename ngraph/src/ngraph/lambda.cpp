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

#include "ngraph/lambda.hpp"
#include "validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr DiscreteTypeInfo Lambda::type_info;

Lambda::Lambda(const OutputVector& results, const ParameterVector& parameters)
    : Lambda(as_result_vector(results), parameters)
{
}

Lambda::Lambda(const ResultVector& results, const ParameterVector& parameters)
    : m_results(results)
    , m_parameters(parameters)
{
}

int64_t Lambda::get_parameter_index(const std::shared_ptr<op::Parameter>& parameter) const
{
    int64_t pos = 0;
    for (auto p : get_parameters())
    {
        if (p == parameter)
        {
            return pos;
        }
        pos++;
    }
    return -1;
}

int64_t Lambda::get_result_index(const Output<Node>& value) const
{
    int64_t pos = 0;
    if (is_type<op::Result>(value.get_node_shared_ptr()))
    {
        auto result = value.get_node_shared_ptr();
        for (auto r : get_results())
        {
            if (r == result)
            {
                return pos;
            }
            pos++;
        }
    }
    else
    {
        for (auto r : get_results())
        {
            if (r->input_value(0) == value)
            {
                return pos;
            }
            pos++;
        }
    }
    return -1;
}

bool Lambda::evaluate(const HostTensorVector& output_tensors, const HostTensorVector& input_tensors)
{
    std::map<RawNodeOutput, HostTensorPtr> value_map;
    for (size_t i = 0; i < m_parameters.size(); ++i)
    {
        value_map[m_parameters.at(i)->output(0)] = input_tensors.at(i);
    }
    OutputVector outputs;
    std::map<RawNodeOutput, HostTensorPtr> output_tensor_map;
    for (size_t i = 0; i < m_results.size(); ++i)
    {
        auto result = m_results.at(i)->output(0);
        output_tensor_map[result] = output_tensors.at(i);
        outputs.push_back(result);
    }
    evaluate_nodes(value_map, output_tensor_map, outputs);
    return true;
}
