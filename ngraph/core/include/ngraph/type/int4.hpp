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

#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/ngraph_visibility.hpp"

#define ROUND_MODE_TO_NEAREST_EVEN

namespace ngraph
{
    class NGRAPH_API int4
    {
    public:
        constexpr int4()
        {
        }
        int4(int8_t value);

        template <typename I>
        explicit int4(I value)
        {
        }

        std::string to_string() const;
        size_t size() const;
        template <typename T>
        bool operator==(const T& other) const;
        template <typename T>
        bool operator!=(const T& other) const
        {
            return !(*this == other);
        }
        template <typename T>
        bool operator<(const T& other) const;
        template <typename T>
        bool operator<=(const T& other) const;
        template <typename T>
        bool operator>(const T& other) const;
        template <typename T>
        bool operator>=(const T& other) const;
        template <typename T>
        int4 operator+(const T& other) const;
        template <typename T>
        int4 operator+=(const T& other);
        template <typename T>
        int4 operator-(const T& other) const;
        template <typename T>
        int4 operator-=(const T& other);
        template <typename T>
        int4 operator*(const T& other) const;
        template <typename T>
        int4 operator*=(const T& other);
        template <typename T>
        int4 operator/(const T& other) const;
        template <typename T>
        int4 operator/=(const T& other);
        operator int8_t() const;

        int8_t to_bits() const;
        friend std::ostream& operator<<(std::ostream& out, const int4& obj)
        {
            out << static_cast<int8_t>(obj);
            return out;
        }

    private:
        constexpr int4(int8_t x, bool)
        {
        }
        struct int4_t {
            int8_t num : 4;
        };


        int4_t m_value = {0};
    };
}
