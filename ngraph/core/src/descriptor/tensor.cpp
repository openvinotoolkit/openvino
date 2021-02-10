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

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/host_tensor.hpp"

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

void descriptor::Tensor::invalidate_values()
{
    m_upper_value = nullptr;
    m_lower_value = nullptr;
}

void descriptor::Tensor::set_lower_value(const HostTensorPtr& value)
{
    NGRAPH_CHECK(m_partial_shape.same_scheme(value->get_partial_shape()));
    NGRAPH_CHECK(m_element_type == value->get_element_type());
    m_lower_value = value;
}

void descriptor::Tensor::set_upper_value(const HostTensorPtr& value)
{
    NGRAPH_CHECK(m_partial_shape.same_scheme(value->get_partial_shape()));
    NGRAPH_CHECK(m_element_type == value->get_element_type());
    m_upper_value = value;
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

NGRAPH_SUPPRESS_DEPRECATED_START
void descriptor::Tensor::set_name(const string& name)
{
    m_name = name;
}

const std::string& descriptor::Tensor::get_name() const
{
    return m_name;
}
NGRAPH_SUPPRESS_DEPRECATED_END

const std::unordered_set<std::string>& descriptor::Tensor::get_names() const
{
    return m_names;
}

void descriptor::Tensor::set_names(const std::unordered_set<std::string>& names)
{
    m_names = names;
}

ostream& operator<<(ostream& out, const descriptor::Tensor& tensor)
{
    std::string names;
    for (const auto& name : tensor.get_names())
    {
        if (!names.empty())
            names += ", ";
        names += name;
    }
    NGRAPH_SUPPRESS_DEPRECATED_START
    if (names.empty())
        names = tensor.get_name();
    NGRAPH_SUPPRESS_DEPRECATED_END
    out << "Tensor(" << names << ")";
    return out;
}
