// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

struct jit_amx_gemmv_int8_call_args {
    const uint8_t* a_ptr;        // packed W tiles (16 x 64 per block)
    const uint8_t* b_ptr;        // packed X tiles (64 x 4 per block)
    std::size_t a_tile_bytes;    // stride between successive A tiles (16*64)
    std::size_t b_group_bytes;   // stride between successive B tiles (64*4)
    std::size_t k_blocks;        // number of 64-wide chunks in K
    int32_t* c_out;              // [16] s32 accumulators (per M-lane)
    const void* tilecfg;         // palette to load (64 bytes, aligned)
};

class jit_amx_gemmv_int8_t : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    jit_amx_gemmv_int8_t();

    using kernel_fn = void (*)(const jit_amx_gemmv_int8_call_args*);

    kernel_fn get() const { return kernel_; }
    const Xbyak::uint8* code_ptr() const { return code_ptr_; }
    size_t code_size() const { return code_size_; }

    void operator()(const jit_amx_gemmv_int8_call_args* args) const {
        if (kernel_) {
            kernel_(args);
        }
    }

    static const jit_amx_gemmv_int8_t& instance();

protected:
    const char* name() const override { return "jit_amx_gemmv_int8_t"; }
    const char* source_file() const override { return __FILE__; }
    void generate() override;

private:
    kernel_fn kernel_ = nullptr;
    const Xbyak::uint8* code_ptr_ = nullptr;
    size_t code_size_ = 0;
};

} // namespace ov::intel_cpu::x64::gemmv_jit
