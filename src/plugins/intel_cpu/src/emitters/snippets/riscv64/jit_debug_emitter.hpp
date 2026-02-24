// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <utility>

#    include "emitters/plugin/riscv64/jit_emitter.hpp"
#    include "emitters/snippets/common/jit_debug_emitter_base.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_debug_emitter : public ov::intel_cpu::jit_debug_emitter_riscv_base<jit_emitter> {
public:
    using base_t = ov::intel_cpu::jit_debug_emitter_riscv_base<jit_emitter>;
    using EmissionLocation = typename base_t::EmissionLocation;

    jit_debug_emitter(const std::shared_ptr<jit_emitter>& target_emitter,
                      std::shared_ptr<jit_emitter> decorator_emitter,
                      const EmissionLocation& loc)
        : base_t(target_emitter->h,
                 target_emitter->host_isa_,
                 target_emitter->exec_prc_,
                 target_emitter->in_out_type_,
                 target_emitter,
                 std::move(decorator_emitter),
                 loc) {}
};

}  // namespace ov::intel_cpu::riscv64

#endif
