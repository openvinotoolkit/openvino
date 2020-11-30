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

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node.hpp"

using namespace ngraph;
using namespace std;

descriptor::Tensor::Tensor(const element::Type& element_type,
                           const PartialShape& pshape,
                           const std::string& name)
    : m_element_type(element_type)
    , m_shape(pshape.is_static() ? pshape.to_shape() : Shape{})
    , m_partial_shape(pshape)
    , m_name(name)
{
}

descriptor::Tensor::Tensor(const element::Type& element_type,
                           const PartialShape& pshape,
                           Node* node,
                           size_t node_output_number)
    : m_element_type(element_type)
    , m_shape(pshape.is_static() ? pshape.to_shape() : Shape{})
    , m_partial_shape(pshape)
    , m_node(node)
    , m_node_output_number(node_output_number)
{
}

void descriptor::Tensor::set_name(const string& name)
{
    m_name = name;
}

void descriptor::Tensor::set_tensor_type(const element::Type& element_type,
                                         const PartialShape& pshape)
{
    set_element_type(element_type);
    set_partial_shape(pshape);
}

void descriptor::Tensor::set_element_type(const element::Type& element_type)
{
    m_element_type = element_type;
}

void descriptor::Tensor::set_partial_shape(const PartialShape& partial_shape)
{
    m_partial_shape = partial_shape;
    if (m_partial_shape.is_static())
    {
        m_shape = m_partial_shape.to_shape();
    }
    else
    {
        m_shape = Shape{};
    }
}

const Shape& descriptor::Tensor::get_shape() const
{
    if (m_partial_shape.is_static())
    {
        return m_shape;
    }
    else
    {
        throw std::invalid_argument(
            "get_shape was called on a descriptor::Tensor with dynamic shape");
    }
}

size_t descriptor::Tensor::size() const
{
    return shape_size(get_shape()) * m_element_type.size();
}

const std::string& descriptor::Tensor::get_name() const
{
    return m_name;
}

ostream& operator<<(ostream& out, const descriptor::Tensor& tensor)
{
    out << "Tensor(" << tensor.get_name() << ")";
    return out;
}
