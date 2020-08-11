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

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;

descriptor::layout::DenseTensorLayout::DenseTensorLayout(const Tensor& tensor)
    : TensorLayout(tensor)
{
}

size_t descriptor::layout::DenseTensorLayout::get_index_offset(const std::vector<size_t>& indices)
{
    auto strides = get_strides();
    if (indices.size() != strides.size())
    {
        throw ngraph_error("Indices have the incorrect rank.");
    }
    size_t result = 0;
    for (uint64_t i = 0; i < indices.size(); i++)
    {
        result += strides[i] * indices[i];
    }
    return result;
}

Strides descriptor::layout::DenseTensorLayout::get_strides() const
{
    return ngraph::row_major_strides(get_shape());
}

bool descriptor::layout::DenseTensorLayout::operator==(const TensorLayout& other) const
{
    const DenseTensorLayout* p_other = dynamic_cast<const DenseTensorLayout*>(&other);
    if (nullptr == p_other)
        return false;

    if (get_element_type() != p_other->get_element_type())
        return false;

    if (get_strides() != p_other->get_strides())
        return false;

    if (get_offset() != p_other->get_offset())
        return false;

    return true;
}
