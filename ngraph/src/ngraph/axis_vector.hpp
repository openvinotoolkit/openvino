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

#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// \brief A vector of axes.
    class AxisVector : public std::vector<size_t>
    {
    public:
        NGRAPH_API AxisVector(const std::initializer_list<size_t>& axes);

        NGRAPH_API AxisVector(const std::vector<size_t>& axes);

        NGRAPH_API AxisVector(const AxisVector& axes);

        NGRAPH_API explicit AxisVector(size_t n);

        template <class InputIterator>
        AxisVector(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        NGRAPH_API AxisVector();

        NGRAPH_API ~AxisVector();

        NGRAPH_API AxisVector& operator=(const AxisVector& v);

        NGRAPH_API AxisVector& operator=(AxisVector&& v) noexcept;
    };

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const AxisVector& axis_vector);
}
