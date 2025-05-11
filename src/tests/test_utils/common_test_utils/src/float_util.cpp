// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/float_util.hpp"

#include "openvino/runtime/exception.hpp"
#include "precomp.hpp"

namespace ov {
namespace test {
namespace utils {

std::string bfloat16_to_bits(bfloat16 f) {
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
    for (int i = 12; i < 16; i += 4) {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

std::string float16_to_bits(float16 f) {
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
    for (int i = 8; i < 16; i += 4) {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

std::string float_to_bits(float f) {
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
    for (int i = 12; i < 32; i += 4) {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

std::string double_to_bits(double d) {
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
    for (int i = 12; i < 64; i += 4) {
        formatted.push_back(' ');
        formatted.append(unformatted, i, 4);
    }
    return formatted;
}

ov::bfloat16 bits_to_bfloat16(const std::string& s) {
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace), unformatted.end());

    if (unformatted.size() != 16) {
        OPENVINO_THROW("Input length must be 16");
    }
    std::bitset<16> bs(unformatted);
    return ov::bfloat16::from_bits(static_cast<uint16_t>(bs.to_ulong()));
}

ov::float16 bits_to_float16(const std::string& s) {
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace), unformatted.end());

    if (unformatted.size() != 16) {
        OPENVINO_THROW("Input length must be 16");
    }
    std::bitset<16> bs(unformatted);
    return ov::float16::from_bits(static_cast<uint16_t>(bs.to_ulong()));
}

float bits_to_float(const std::string& s) {
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace), unformatted.end());

    if (unformatted.size() != 32) {
        OPENVINO_THROW("Input length must be 32");
    }
    std::bitset<32> bs(unformatted);
    FloatUnion fu;
    fu.i = static_cast<uint32_t>(bs.to_ulong());
    return fu.f;
}

double bits_to_double(const std::string& s) {
    std::string unformatted = s;
    unformatted.erase(remove_if(unformatted.begin(), unformatted.end(), ::isspace), unformatted.end());

    if (unformatted.size() != 64) {
        OPENVINO_THROW("Input length must be 64");
    }
    std::bitset<64> bs(unformatted);
    DoubleUnion du;
    du.i = static_cast<uint64_t>(bs.to_ullong());
    return du.d;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
