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

#include <algorithm>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/node.hpp"

using namespace std;
using namespace ngraph;

descriptor::Output::Output(Node* node, size_t index, const shared_ptr<Tensor>& tensor)
    : m_node(node)
    , m_index(index)
    , m_tensor(tensor)
{
}

// Add an input to the vector of inputs that use this output.
void descriptor::Output::add_input(Input* input)
{
    // Keep the inputs in insertion order to keep sorts deterministic
    if (find(m_inputs.begin(), m_inputs.end(), input) == m_inputs.end())
    {
        m_inputs.push_back(input);
    }
}

void descriptor::Output::remove_input(Input* input)
{
    auto it = find(m_inputs.begin(), m_inputs.end(), input);
    if (it != m_inputs.end())
    {
        m_inputs.erase(it);
    }
}

shared_ptr<Node> descriptor::Output::get_node() const
{
    return m_node->shared_from_this();
}

ngraph::Output<Node> descriptor::Output::get_output() const
{
    return get_node()->output(m_index);
}

descriptor::Tensor& descriptor::Output::get_tensor() const
{
    return *m_tensor;
}

const Shape& descriptor::Output::get_shape() const
{
    return m_tensor->get_shape();
}

const PartialShape& descriptor::Output::get_partial_shape() const
{
    return m_tensor->get_partial_shape();
}

const element::Type& descriptor::Output::get_element_type() const
{
    return m_tensor->get_element_type();
}
