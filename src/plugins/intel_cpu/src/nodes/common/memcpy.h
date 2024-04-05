// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include "node.h"
// #include "cpu/x64/jit_generator.hpp"
// #include "common/utils.hpp"
// #include "cpu/x64/cpu_isa_traits.hpp"

// using namespace dnnl;
// using namespace dnnl::impl;
// using namespace dnnl::impl::cpu::x64;
// using namespace dnnl::impl::utils;

#include <cassert>
#include <cstdint>
#include <memory>

namespace ov {
namespace intel_cpu {

struct jit_args_memcpy {
    const void* src;
    const void* dst;
    std::size_t size;
};

struct jit_uni_memcpy_kernel {
    void (*ker_)(const jit_args_memcpy *);

    void operator()(const jit_args_memcpy *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_memcpy_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_memcpy_kernel() {}

    virtual void create_ker() = 0;
};

class MemCpy {
public:
    MemCpy();

    void execute(const uint8_t* src_data, uint8_t* dst_data, std::size_t size);

private:
    std::shared_ptr<jit_uni_memcpy_kernel> memcpy_kernel;
};

}   // namespace intel_cpu
}   // namespace ov
