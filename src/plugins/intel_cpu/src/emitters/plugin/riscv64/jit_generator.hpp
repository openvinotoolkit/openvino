// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>

#include "xbyak_riscv/xbyak_riscv.hpp"
#include "../src/common/c_types_map.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

class jit_generator : public Xbyak_riscv::CodeGenerator {
public:
    const uint8_t *jit_ker() const {
        assert(jit_ker_ && "jit_ker_ is nullable");
        return jit_ker_;
    }

    void preamble();
    void postamble();

    void L(const char *label) = delete;
    void L(Xbyak_riscv::Label &label) {
        Xbyak_riscv::CodeGenerator::L(label);
    }

    virtual void create_kernel();

protected:
    virtual void generate() = 0;
    const uint8_t *jit_ker_ = nullptr;

    static inline bool is_initialized() {
        /* At the moment, Xbyak_aarch64 does not have GetError()\
         so that return dummy result. */
        return true;
    }

private:
    const uint8_t* getCode();
};

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
