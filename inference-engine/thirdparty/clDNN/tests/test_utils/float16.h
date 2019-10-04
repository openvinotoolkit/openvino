/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once
#include "include/math_utils.h"

struct FLOAT16
{
    struct representation
    {
        uint16_t significand : 10;
        uint16_t exponent : 5;
        uint16_t sign : 1;
    };

    union
    {
        uint16_t v = 0;
        representation format; // added this struct for the .natvis file (for debug)
    };

    static FLOAT16 min_val()
    {
        FLOAT16 f16;
        f16.v = 0xFC00;
        return f16;
    }

    operator double() const { double d = (double)float16_to_float32(v); return d; }
    operator float() const { float f = float16_to_float32(v); return f; }
    operator int16_t() const { return *(int16_t*)(&v); }
    operator uint32_t() const { return v; }
    FLOAT16(float f) { v = float32_to_float16(f); }
    FLOAT16(int i) { v = float32_to_float16(float(i)); }
    explicit FLOAT16(int16_t d) : v(d) {}
    friend FLOAT16 operator +(const FLOAT16 &v1, const FLOAT16 &v2);
    friend FLOAT16 operator -(const FLOAT16 &v1, const FLOAT16 &v2);
    friend FLOAT16 operator *(const FLOAT16 &v1, const FLOAT16 &v2);
    friend FLOAT16 operator /(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator >(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator >=(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator <(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator >(const FLOAT16 &v1, const float &v2);
    friend bool operator <(const FLOAT16 &v1, const float &v2);
    friend bool operator !=(const FLOAT16 &v1, const FLOAT16 &v2);

    FLOAT16() {}

    FLOAT16& operator +=(const FLOAT16 &v1)
    {
            *this = (float)*this + (float)v1;
            return *this;
    }

    FLOAT16& operator /=(const FLOAT16 &v1)
    {
            *this = (float)*this / (float)v1;
            return *this;
    }

    FLOAT16& operator *=(const FLOAT16 &v1)
    {
        *this = (float)*this * (float)v1;
        return *this;
    }
};

inline FLOAT16 operator +(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return (float)v1 + (float)v2;
}

inline FLOAT16 operator -(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return (float)v1 - (float)v2;
}

inline FLOAT16 operator *(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return (float)v1 * (float)v2;
}

inline FLOAT16 operator /(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return (float)v1 / (float)v2;
}

inline bool operator >(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return (float)v1 > (float)v2;
}

inline bool operator >=(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return (float)v1 >= (float)v2;
}

inline bool operator <(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return (float)v1 < (float)v2;
}

inline bool operator >(const FLOAT16 &v1, const float &v2)
{
    return (float)v1 > v2;
}

inline bool operator <(const FLOAT16 &v1, const float &v2)
{
    return (float)v1 < v2;
}

inline bool operator !=(const FLOAT16 &v1, const FLOAT16 &v2)
{
    return v1.v != v2.v;
}