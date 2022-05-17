/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020-2021 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>

#include "cpu/aarch64/cpu_barrier.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace simple_barrier {

void generate(jit_generator &code, Xbyak_aarch64::XReg reg_ctx,
        Xbyak_aarch64::XReg reg_nthr, bool usedAsFunc) {
#define BAR_CTR_OFF offsetof(ctx_t, ctr)
#define BAR_SENSE_OFF offsetof(ctx_t, sense)
    using namespace Xbyak_aarch64;

    const XReg x_tmp_0 = (usedAsFunc) ? code.x9 : code.X_TMP_0;
    const WReg w_tmp_1 = (usedAsFunc) ? code.w10 : code.W_TMP_1;
    const XReg x_addr_sense = (usedAsFunc) ? code.x11 : code.X_TMP_2;
    const XReg x_addr_ctx = (usedAsFunc) ? code.x12 : code.X_TMP_3;
    const XReg x_sense = (usedAsFunc) ? code.x13 : code.X_TMP_4;
    const XReg x_tmp_addr = (usedAsFunc) ? code.x14 : code.X_DEFAULT_ADDR;

    Label barrier_exit_label, spin_label, atomic_label;

    code.cmp(reg_nthr, 1);
    code.b(EQ, barrier_exit_label);

    /* take and save current sense */
    code.add_imm(x_addr_sense, reg_ctx, BAR_SENSE_OFF, x_tmp_0);
    code.ldr(x_sense, ptr(x_addr_sense));

    code.add_imm(x_addr_ctx, reg_ctx, BAR_CTR_OFF, x_tmp_addr);
    if (mayiuse(sve_512)) {
        code.prfm(PLDL1KEEP, ptr(x_addr_ctx));
        code.prfm(PLDL1KEEP, ptr(x_addr_ctx));
    }

    if (mayiuse_atomic()) {
        code.mov(x_tmp_0, 1);
        code.ldaddal(x_tmp_0, x_tmp_0, ptr(x_addr_ctx));
        code.add(x_tmp_0, x_tmp_0, 1);
    } else {
        code.L(atomic_label);
        code.ldaxr(x_tmp_0, ptr(x_addr_ctx));
        code.add(x_tmp_0, x_tmp_0, 1);
        code.stlxr(w_tmp_1, x_tmp_0, ptr(x_addr_ctx));
        code.cbnz(w_tmp_1, atomic_label);
    }
    code.cmp(x_tmp_0, reg_nthr);
    code.b(NE, spin_label);

    /* the last thread {{{ */
    code.mov_imm(x_tmp_0, 0);
    code.str(x_tmp_0, ptr(x_addr_ctx)); // reset ctx
    /* commit CTX clear, before modify SENSE,
       otherwise other threads load old SENSE value. */
    code.dmb(ISH);

    // notify waiting threads
    code.mvn(x_sense, x_sense);
    code.str(x_sense, ptr(x_addr_sense));
    code.b(barrier_exit_label);
    /* }}} the last thread */

    code.L(spin_label);
    code.yield();
    code.ldr(x_tmp_0, ptr(x_addr_sense));
    code.cmp(x_tmp_0, x_sense);
    code.b(EQ, spin_label);

    code.dmb(ISH);
    code.L(barrier_exit_label);

#undef BAR_CTR_OFF
#undef BAR_SENSE_OFF
}

/** jit barrier generator */
struct jit_t : public jit_generator {

    void generate() override {
        simple_barrier::generate(*this, abi_param1, abi_param2, true);
        ret();
    }

    // TODO: Need to check status
    jit_t() { create_kernel(); }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_t)
};

void barrier(ctx_t *ctx, int nthr) {
    static jit_t j;
    j(ctx, nthr);
}

} // namespace simple_barrier

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
