// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64::utils {

inline size_t get_snippet_lanes() {
    const auto vlen_bytes = Xbyak_riscv::CPU::getInstance().getVlen() / 8;
    OPENVINO_ASSERT(vlen_bytes % sizeof(float) == 0,
                    "Unexpected RVV VLEN in bytes: ",
                    vlen_bytes,
                    ". Snippets expect an integer number of f32 lanes.");
    return vlen_bytes / sizeof(float);
}

inline static std::vector<Xbyak_riscv::Reg> transform_idxs_to_regs(const std::vector<size_t>& idxs) {
    std::vector<Xbyak_riscv::Reg> regs;
    regs.reserve(idxs.size());
    std::transform(idxs.begin(), idxs.end(), std::back_inserter(regs), [](size_t idx) {
        return Xbyak_riscv::Reg(static_cast<int>(idx));
    });
    return regs;
}

/**
 * @brief RAII wrapper for acquiring and managing auxiliary general-purpose registers in JIT compilation.
 * The class supports two allocation strategies:
 * - If a register pool is available, it borrows a register from the pool and returns it upon destruction
 * - If the pool is empty, it manually allocates an available register and preserves its original value
 *   on the stack, restoring it upon destruction
 * This ensures that temporary register usage doesn't interfere with the existing register state
 * and provides safe register management in complex JIT scenarios like loop emitters.
 */
class jit_aux_gpr_holder {
public:
    jit_aux_gpr_holder(ov::intel_cpu::riscv64::jit_generator_t* host,
                       std::vector<size_t>& pool_gpr_idxs,
                       const std::vector<size_t>& used_gpr_idxs);

    ~jit_aux_gpr_holder();

    [[nodiscard]] const Xbyak_riscv::Reg& get_reg() const {
        return m_reg;
    }

private:
    ov::intel_cpu::riscv64::jit_generator_t* m_h;
    std::vector<size_t>& m_pool_gpr_idxs;
    Xbyak_riscv::Reg m_reg;
    bool m_preserved = false;
};

/**
 * @brief Find the available register from the pool excepting: a0, a1, sp, ra and `used_gpr_idxs`
 * @param used_gpr_idxs current used gpr register indexes
 * @return register
 */
Xbyak_riscv::Reg get_aux_gpr(const std::vector<size_t>& used_gpr_idxs);

}  // namespace ov::intel_cpu::riscv64::utils
