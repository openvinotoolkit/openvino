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

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// \brief Strides for a tensor.
    class Strides : public std::vector<size_t>
    {
    public:
        NGRAPH_API Strides();

        NGRAPH_API Strides(const std::initializer_list<size_t>& axis_strides);

        NGRAPH_API Strides(const std::vector<size_t>& axis_strides);

        NGRAPH_API Strides(const Strides& axis_strides);

        NGRAPH_API explicit Strides(size_t n, size_t initial_value = 0);

        template <class InputIterator>
        Strides(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        NGRAPH_API Strides& operator=(const Strides& v);

        NGRAPH_API Strides& operator=(Strides&& v) noexcept;
    };

    template <>
    class NGRAPH_API AttributeAdapter<Strides> : public ValueReference<Strides>,
                                                 public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(Strides& value)
            : ValueReference<Strides>(value)
        {
        }
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<Strides>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const Strides& strides);
}
