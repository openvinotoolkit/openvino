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

#include <algorithm>
#include <bitset>
#include <cmath>
#include <limits>
#include <sstream>

#include "ngraph/ngraph.hpp"

namespace ngraph
{
    namespace test
    {
        union FloatUnion {
            FloatUnion() { i = 0; }
            FloatUnion(float val) { f = val; }
            FloatUnion(uint32_t val) { i = val; }
            FloatUnion(uint32_t s, uint32_t e, uint32_t f)
                : FloatUnion(s << 31 | e << 23 | f)
            {
            }
            float f;
            uint32_t i;
        };

        union DoubleUnion {
            DoubleUnion() { i = 0; }
            DoubleUnion(double val) { d = val; }
            DoubleUnion(uint64_t val) { i = val; }
            double d;
            uint64_t i;
        };

        std::string bfloat16_to_bits(bfloat16 f);

        std::string float16_to_bits(float16 f);

        std::string float_to_bits(float f);

        std::string double_to_bits(double d);

        bfloat16 bits_to_bfloat16(const std::string& s);

        float bits_to_float(const std::string& s);

        double bits_to_double(const std::string& s);

        float16 bits_to_float16(const std::string& s);
    }
}
