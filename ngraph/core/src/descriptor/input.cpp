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

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace descriptor;

descriptor::Input::Input(Node* node, size_t index, Output& output)
    : m_node(node)
    , m_index(index)
    , m_output(&output)
    , m_is_relevant_to_shape(false)
    , m_is_relevant_to_value(true)
{
    m_src_node = std::shared_ptr<Node>(output.get_node());
    output.add_input(this);
}

descriptor::Input::Input(Node* node, size_t index)
    : m_node(node)
    , m_index(index)
    , m_output(nullptr)
    , m_is_relevant_to_shape(false)
    , m_is_relevant_to_value(true)
{
}

descriptor::Input::~Input()
{
    remove_output();
}

void descriptor::Input::replace_output(Output& new_output)
{
    if (m_output != nullptr)
    {
        m_output->remove_input(this);
    }
    new_output.add_input(this);
    m_output = &new_output;
    m_src_node = std::shared_ptr<Node>(new_output.get_node());

    if (getenv_bool("NGRAPH_ENABLE_REPLACE_CHECK"))
    {
        // the result of clone_with_new_inputs will be thrown away or
        // an exception will be thrown by `m_node`'s class c-tor
        // if a new input violates one of the type checks in the c-tor.
        m_node->clone_with_new_inputs(m_node->input_values());
    }
}

void descriptor::Input::replace_output(std::shared_ptr<Node> node, size_t i)
{
    replace_output(node->m_outputs.at(i));
}

void descriptor::Input::remove_output()
{
    if (m_output != nullptr)
    {
        m_output->remove_input(this);
        m_src_node = nullptr;
        m_output = nullptr;
    }
}

std::shared_ptr<Node> descriptor::Input::get_node() const
{
    return m_node->shared_from_this();
}

const Tensor& descriptor::Input::get_tensor() const
{
    return m_output->get_tensor();
}

Tensor& descriptor::Input::get_tensor()
{
    return m_output->get_tensor();
}

std::shared_ptr<const Tensor> descriptor::Input::get_tensor_ptr() const
{
    return m_output->get_tensor_ptr();
}

std::shared_ptr<Tensor> descriptor::Input::get_tensor_ptr()
{
    return m_output->get_tensor_ptr();
}

const Shape& descriptor::Input::get_shape() const
{
    return m_output->get_shape();
}

const PartialShape& descriptor::Input::get_partial_shape() const
{
    return m_output->get_partial_shape();
}

const element::Type& descriptor::Input::get_element_type() const
{
    return m_output->get_element_type();
}
