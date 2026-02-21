// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu/x64/jit_generator.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

struct jit_amx_gemmv_bf16_call_args {
    const uint16_t* a_ptr;
    const uint16_t* b_ptr;
    std::size_t a_tile_bytes;
    std::size_t b_group_bytes;
    std::size_t k_blocks;
    float* c_out;
};

class jit_amx_gemmv_bf16_t : public dnnl::impl::cpu::x64::jit_generator_t {
public:
    jit_amx_gemmv_bf16_t();

    using kernel_fn = void (*)(const jit_amx_gemmv_bf16_call_args*);

    kernel_fn get() const { return kernel_; }

    void operator()(const jit_amx_gemmv_bf16_call_args* args) const {
        if (kernel_) {
            kernel_(args);
        }
    }

    static const jit_amx_gemmv_bf16_t& instance();

protected:
    const char* name() const override { return "jit_amx_gemmv_bf16_t"; }
    const char* source_file() const override { return __FILE__; }
    void generate() override;

private:
    kernel_fn kernel_ = nullptr;
};

} // namespace ov::intel_cpu::x64::gemmv_jit
