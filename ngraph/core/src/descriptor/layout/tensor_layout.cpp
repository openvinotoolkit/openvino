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

#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;

descriptor::layout::TensorLayout::TensorLayout(const descriptor::Tensor& tensor)
    : m_element_type(tensor.get_element_type())
    , m_shape(tensor.get_shape())
{
}

const element::Type& descriptor::layout::TensorLayout::get_element_type() const
{
    return m_element_type;
}

const Shape& descriptor::layout::TensorLayout::get_shape() const
{
    return m_shape;
}

size_t descriptor::layout::TensorLayout::get_size() const
{
    return ngraph::shape_size(get_shape());
}

size_t descriptor::layout::TensorLayout::get_allocated_size()
{
    return get_size() * get_element_type().size();
}
