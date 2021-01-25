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

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Result::type_info;

op::Result::Result(const Output<Node>& arg, bool needs_default_layout)
    : Op({arg})
    , m_needs_default_layout(needs_default_layout)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Result::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Result_visit_attributes);
    return true;
}

void op::Result::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_Result_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this, get_input_size() == 1, "Argument has ", get_input_size(), " outputs (1 expected).");

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::Result::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Result_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    auto res = make_shared<Result>(new_args.at(0), m_needs_default_layout);
    return std::move(res);
}

bool op::Result::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Result_evaluate);
    outputs[0]->set_unary(inputs[0]);
    void* output = outputs[0]->get_data_ptr();
    void* input = inputs[0]->get_data_ptr();
    memcpy(output, input, outputs[0]->get_size_in_bytes());
    return true;
}

bool op::Result::constant_fold(OutputVector& output_values, const OutputVector& inputs_values)
{
    return false;
}

constexpr DiscreteTypeInfo AttributeAdapter<ResultVector>::type_info;

AttributeAdapter<ResultVector>::AttributeAdapter(ResultVector& ref)
    : m_ref(ref)
{
}

bool AttributeAdapter<ResultVector>::visit_attributes(AttributeVisitor& visitor)
{
    int64_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size())
    {
        m_ref.resize(size);
    }
    ostringstream index;
    for (int64_t i = 0; i < size; i++)
    {
        index.str("");
        index << i;
        string id;
        if (m_ref[i])
        {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i])
        {
            m_ref[i] = as_type_ptr<op::v0::Result>(visitor.get_registered_node(id));
        }
    }
    return true;
}
