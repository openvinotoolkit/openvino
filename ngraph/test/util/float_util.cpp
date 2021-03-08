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

#include "util/float_util.hpp"

std::string ngraph::test::bfloat16_to_bits(bfloat16 f)
{
    std::stringstream ss;
    ss << std::bitset<16>(f.to_bits());
    std::string unformatted = ss.str();
    std::string formatted;
    formatted.reserve(41);
    // Sign
    formatted.push_back(unformatted[0]);
    formatted.append("  ");
    // Exponent
    formatted.append(unformatted, 1, 8);
    formatted.append("  ");
    // Mantissa
    formatted.append(unformatted, 9, 3);
    for (int i = 12; i < 16; i += 4)
    {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

std::string ngraph::test::float16_to_bits(float16 f)
{
    std::stringstream ss;
    ss << std::bitset<16>(f.to_bits());
    std::string unformatted = ss.str();
    std::string formatted;
    formatted.reserve(41);
    // Sign
    formatted.push_back(unformatted[0]);
    formatted.append("  ");
    // Exponent
    formatted.append(unformatted, 1, 5);
    formatted.append("  ");
    // Mantissa
    formatted.append(unformatted, 6, 2);
    for (int i = 8; i < 16; i += 4)
    {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

std::string ngraph::test::float_to_bits(float f)
{
    FloatUnion fu{f};
    std::stringstream ss;
    ss << std::bitset<32>(fu.i);
    std::string unformatted = ss.str();
    std::string formatted;
    formatted.reserve(41);
    // Sign
    formatted.push_back(unformatted[0]);
    formatted.append("  ");
    // Exponent
    formatted.append(unformatted, 1, 8);
    formatted.append("  ");
    // Mantissa
    formatted.append(unformatted, 9, 3);
    for (int i = 12; i < 32; i += 4)
    {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

std::string ngraph::test::double_to_bits(double d)
{
    DoubleUnion du{d};
    std::stringstream ss;
    ss << std::bitset<64>(du.i);
    std::string unformatted = ss.str();
    std::string formatted;
    formatted.reserve(80);
    // Sign
    formatted.push_back(unformatted[0]);
    formatted.append("  ");
    // Exponent
    formatted.append(unformatted, 1, 11);
    formatted.push_back(' ');
    // Mantissa
    for (int i = 12; i < 64; i += 4)
    {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

ngraph::bfloat16 ngraph::test::bits_to_bfloat16(const std::string& s)
{
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace),
                      unformatted.end());

    if (unformatted.size() != 16)
    {
        throw ngraph_error("Input length must be 16");
    }
    std::bitset<16> bs(unformatted);
    return bfloat16::from_bits(static_cast<uint16_t>(bs.to_ulong()));
}

ngraph::float16 ngraph::test::bits_to_float16(const std::string& s)
{
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace),
                      unformatted.end());

    if (unformatted.size() != 16)
    {
        throw ngraph_error("Input length must be 16");
    }
    std::bitset<16> bs(unformatted);
    return float16::from_bits(static_cast<uint16_t>(bs.to_ulong()));
}

float ngraph::test::bits_to_float(const std::string& s)
{
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace),
                      unformatted.end());

    if (unformatted.size() != 32)
    {
        throw ngraph_error("Input length must be 32");
    }
    std::bitset<32> bs(unformatted);
    FloatUnion fu;
    fu.i = static_cast<uint32_t>(bs.to_ulong());
    return fu.f;
}

double ngraph::test::bits_to_double(const std::string& s)
{
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace),
                      unformatted.end());

    if (unformatted.size() != 64)
    {
        throw ngraph_error("Input length must be 64");
    }
    std::bitset<64> bs(unformatted);
    DoubleUnion du;
    du.i = static_cast<uint64_t>(bs.to_ullong());
    return du.d;
}
