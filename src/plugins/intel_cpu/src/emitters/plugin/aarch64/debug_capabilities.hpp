#ifndef DEBUG_CAPABILITIES_HPP
#define DEBUG_CAPABILITIES_HPP

#include <iostream>
#include <cstdint>
#include <arm_neon.h> // For SIMD support
#include "openvino/util/ov_string_utils.hpp" // For ov::util::join

class RegPrints {
public:
    // Print general-purpose registers
    static void print_gpr(const uint64_t& reg_value, const char* reg_name) {
        std::cout << "Register " << reg_name << ": " << std::hex << reg_value << std::endl;
    }

    // Print vector registers (SIMD)
    static void print_simd(const float32x4_t& reg_value, const char* reg_name) {
        float values[4];
        vst1q_f32(values, reg_value); // Store SIMD register into an array
        std::cout << "SIMD Register " << reg_name << ": ["
                  << ov::util::join(values, values + 4, ", ") << "]" << std::endl;
    }
};

#endif // DEBUG_CAPABILITIES_HPP
