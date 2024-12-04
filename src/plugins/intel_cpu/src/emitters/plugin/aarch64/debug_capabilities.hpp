#ifndef DEBUG_CAPABILITIES_HPP
#define DEBUG_CAPABILITIES_HPP

#include <iostream>
#include <cstdint>
#include <arm_neon.h> // For SIMD support
#include "openvino/util/ov_string_utils.hpp" // For ov::util::join

class RegPrints {
public:
    static void print_gpr(jit_generator &gen, const uint64_t &reg_value, const char *reg_name) {
        // Emit JIT code to print general-purpose register during runtime
        gen.mov(gen.rdi, reg_value); // Move register value into rdi
        gen.mov(gen.rsi, reinterpret_cast<uint64_t>(reg_name)); // Pass register name as argument
        gen.call(reinterpret_cast<void(*)(const char*, uint64_t)>(print_runtime_gpr)); // Call runtime function
    }

    static void print_simd(jit_generator &gen, const float32x4_t &reg_value, const char *reg_name) {
        // Emit JIT code to handle SIMD printing during runtime
        gen.mov(gen.rdi, reinterpret_cast<uint64_t>(&reg_value)); // Move SIMD value into rdi
        gen.mov(gen.rsi, reinterpret_cast<uint64_t>(reg_name)); // Pass register name as argument
        gen.call(reinterpret_cast<void(*)(const char*, const float*)>(print_runtime_simd)); // Call runtime function
    }

private:
    // Runtime functions to print the registers
    static void print_runtime_gpr(const char *reg_name, uint64_t value) {
        std::cout << "Register " << reg_name << ": " << std::hex << value << std::endl;
    }

    static void print_runtime_simd(const char *reg_name, const float *values) {
        std::cout << "SIMD Register " << reg_name << ": [" 
                  << values[0] << ", " << values[1] << ", " << values[2] << ", " << values[3] << "]" << std::endl;
    }
};

#endif // DEBUG_CAPABILITIES_HPP
