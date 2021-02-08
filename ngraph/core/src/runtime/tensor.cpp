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

#include "ngraph/runtime/tensor.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace std;

const Shape& runtime::Tensor::get_shape() const
{
    return m_descriptor->get_shape();
}

const PartialShape& runtime::Tensor::get_partial_shape() const
{
    return m_descriptor->get_partial_shape();
}

const element::Type& runtime::Tensor::get_element_type() const
{
    return m_descriptor->get_element_type();
}

size_t runtime::Tensor::get_element_count() const
{
    return shape_size(m_descriptor->get_shape());
}

size_t runtime::Tensor::get_size_in_bytes() const
{
    return m_descriptor->size();
}

const std::string& runtime::Tensor::get_name() const
{
    return m_descriptor->get_name();
}

bool runtime::Tensor::get_stale() const
{
    return m_stale;
}

void runtime::Tensor::set_stale(bool val)
{
    m_stale = val;
}
