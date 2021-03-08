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

#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<float>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<double>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<string>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<bool>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<int8_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<int16_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<int32_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<int64_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<uint8_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<uint16_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<uint32_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<uint64_t>::type_info;
#ifdef __APPLE__
    // size_t is not uint_64t on OSX
    constexpr DiscreteTypeInfo AttributeAdapter<size_t>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<size_t>>::type_info;
#endif
    constexpr DiscreteTypeInfo AttributeAdapter<vector<int8_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<int16_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<int32_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<int64_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<uint8_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<uint16_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<uint32_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<uint64_t>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<float>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<double>>::type_info;
    constexpr DiscreteTypeInfo AttributeAdapter<vector<string>>::type_info;
}
