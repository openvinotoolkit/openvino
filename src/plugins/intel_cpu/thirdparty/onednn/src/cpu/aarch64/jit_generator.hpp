/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_GENERATOR_HPP
#define CPU_AARCH64_JIT_GENERATOR_HPP

#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#include "cpu/jit_utils/jit_utils.hpp"

#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return STRINGIFY(jit_name); } \
    const char *source_file() const override { return __FILE__; }

static const size_t CSIZE = sizeof(uint32_t);

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// TODO: move this to jit_generator class?
namespace {

typedef enum {
    MAX_CODE_SIZE = 256 * 1024,
} max_code_size_t;

// Callee-saved registers
constexpr Xbyak_aarch64::Operand::Code abi_save_gpr_regs[]
        = {Xbyak_aarch64::Operand::X19, Xbyak_aarch64::Operand::X20,
                Xbyak_aarch64::Operand::X21, Xbyak_aarch64::Operand::X22,
                Xbyak_aarch64::Operand::X23, Xbyak_aarch64::Operand::X24,
                Xbyak_aarch64::Operand::X25, Xbyak_aarch64::Operand::X26,
                Xbyak_aarch64::Operand::X27, Xbyak_aarch64::Operand::X28};

// See "Procedure Call Standsard for the ARM 64-bit Architecture (AArch64)"
static const Xbyak_aarch64::XReg abi_param1(Xbyak_aarch64::Operand::X0),
        abi_param2(Xbyak_aarch64::Operand::X1),
        abi_param3(Xbyak_aarch64::Operand::X2),
        abi_param4(Xbyak_aarch64::Operand::X3),
        abi_param5(Xbyak_aarch64::Operand::X4),
        abi_param6(Xbyak_aarch64::Operand::X5),
        abi_param7(Xbyak_aarch64::Operand::X6),
        abi_param8(Xbyak_aarch64::Operand::X7),
        abi_not_param1(Xbyak_aarch64::Operand::X15);
} // namespace

class jit_generator : public Xbyak_aarch64::CodeGenerator, public c_compatible {
public:
    using c_compatible::operator new;
    using c_compatible::operator new[];
    using c_compatible::operator delete;
    using c_compatible::operator delete[];

private:
    const size_t xreg_len = 8;
    const size_t vreg_len_preserve = 8; // Only bottom 8byte must be preserved.
    const size_t vreg_to_preserve = 8; // VREG8 - VREG15

    const size_t num_abi_save_gpr_regs
            = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t preserved_stack_size = xreg_len * (2 + num_abi_save_gpr_regs)
            + vreg_len_preserve * vreg_to_preserve;

    const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * x0.getBit() / 8
            + vreg_to_preserve * vreg_len_preserve;

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    const Xbyak_aarch64::WReg W_TMP_0 = w23;
    const Xbyak_aarch64::WReg W_TMP_1 = w24;
    const Xbyak_aarch64::WReg W_TMP_2 = w25;
    const Xbyak_aarch64::WReg W_TMP_3 = w26;
    const Xbyak_aarch64::WReg W_TMP_4 = w27;
    const Xbyak_aarch64::XReg X_TMP_0 = x23;
    const Xbyak_aarch64::XReg X_TMP_1 = x24;
    const Xbyak_aarch64::XReg X_TMP_2 = x25;
    const Xbyak_aarch64::XReg X_TMP_3 = x26;
    const Xbyak_aarch64::XReg X_TMP_4 = x27;
    const Xbyak_aarch64::XReg X_DEFAULT_ADDR = x28;
    const Xbyak_aarch64::XReg X_SP = x21;
    const Xbyak_aarch64::XReg X_TRANSLATOR_STACK = x22;
    const Xbyak_aarch64::PReg P_TMP = p0;
    const Xbyak_aarch64::PReg P_TMP_0 = p11;
    const Xbyak_aarch64::PReg P_TMP_1 = p12;
    const Xbyak_aarch64::PReg P_ALL_ZERO = p10;
    const Xbyak_aarch64::PReg P_NOT_256 = p13;
    const Xbyak_aarch64::PReg P_NOT_128 = p14;
    const Xbyak_aarch64::PReg P_ALL_ONE = p15;

    const std::vector<Xbyak_aarch64::XReg> x_tmp_vec
            = {X_TMP_0, X_TMP_1, X_TMP_2, X_TMP_3, X_TMP_4};
    const int x_tmp_vec_size = x_tmp_vec.size();

    const Xbyak_aarch64::XReg param1 = abi_param1;
    constexpr static size_t translator_stack_offset = 1024 * 128;
    constexpr static uint32_t DUMMY_IDX = 99;

    inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

    void preamble() {
        using namespace Xbyak_aarch64::util;
        uint64_t sveLen = get_sve_length();

        stp(x29, x30, pre_ptr(sp, -16));
        /* x29 is a frame pointer. */
        mov(x29, sp);
        sub(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);

        /* x9 can be used as a temporal register. */
        mov(x9, sp);

        if (vreg_to_preserve) {
            st4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
            st4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
        }
        for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
            stp(Xbyak_aarch64::XReg(abi_save_gpr_regs[i]),
                    Xbyak_aarch64::XReg(abi_save_gpr_regs[i + 1]),
                    post_ptr(x9, xreg_len * 2));
        }

        if (sveLen) { /* SVE is available. */
            ptrue(P_ALL_ONE.b);
            pfalse(P_ALL_ZERO.b);
        }
        if (sveLen >= SVE_256) {
            ptrue(P_NOT_128.b, Xbyak_aarch64::VL16);
            not_(P_NOT_128.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_128.b);
        }
        if (sveLen >= SVE_512) {
            ptrue(P_NOT_256.b, Xbyak_aarch64::VL32);
            not_(P_NOT_256.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_NOT_256.b);
        }

        mov(X_SP, sp);
        sub_imm(X_TRANSLATOR_STACK, X_SP, translator_stack_offset, X_TMP_0);
    }

    void postamble() {
        using namespace Xbyak_aarch64::util;
        uint64_t sveLen = get_sve_length();

        mov(x9, sp);

        if (sveLen) /* SVE is available. */
            eor(P_ALL_ONE.b, P_ALL_ONE / Xbyak_aarch64::T_z, P_ALL_ONE.b,
                    P_ALL_ONE.b);
        if (sveLen >= SVE_256)
            eor(P_NOT_128.b, P_NOT_128 / Xbyak_aarch64::T_z, P_NOT_128.b,
                    P_NOT_128.b);
        if (sveLen >= SVE_512)
            eor(P_NOT_256.b, P_NOT_256 / Xbyak_aarch64::T_z, P_NOT_256.b,
                    P_NOT_256.b);

        if (vreg_to_preserve) {
            ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_len_preserve * 4));
            ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_len_preserve * 4));
        }

        for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
            ldp(Xbyak_aarch64::XReg(abi_save_gpr_regs[i]),
                    Xbyak_aarch64::XReg(abi_save_gpr_regs[i + 1]),
                    post_ptr(x9, xreg_len * 2));
        }

        add(sp, sp, static_cast<int64_t>(preserved_stack_size) - 16);
        ldp(x29, x30, post_ptr(sp, 16));
        ret();
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak_aarch64::Label &label) {
        Xbyak_aarch64::CodeGenerator::L(label);
    }

    void L_aligned(Xbyak_aarch64::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    void uni_clear(const Xbyak_aarch64::VReg &dst) { eor(dst.b, dst.b, dst.b); }

    void uni_clear(const Xbyak_aarch64::ZReg &dst) { eor(dst.d, dst.d, dst.d); }

    template <typename TReg>
    void uni_fdiv(const TReg &dst, const TReg &src, const TReg &src2) {
        fdiv(dst, src, src2);
    }

    void uni_fdiv(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2,
            const Xbyak_aarch64::VReg4S &tmp, const Xbyak_aarch64::PReg &pred) {
        UNUSED(tmp);
        UNUSED(pred);
        fdiv(dst, src, src2);
    }

    template <typename TReg>
    void uni_fdiv(const TReg &dst, const TReg &src, const TReg &src2,
            const TReg &tmp, const Xbyak_aarch64::PReg &pred) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        uint32_t src2Idx = src2.getIdx();
        uint32_t tmpIdx = tmp.getIdx();

        if (dstIdx == src2Idx) {
            assert(tmpIdx != srcIdx && tmpIdx != src2Idx);

            mov(Xbyak_aarch64::ZRegD(tmpIdx), Xbyak_aarch64::ZRegD(src2Idx));
            mov(dst, pred / Xbyak_aarch64::T_m, src);
            fdiv(dst, pred / Xbyak_aarch64::T_m, tmp);
        } else if (dstIdx == srcIdx) {
            fdiv(dst, pred / Xbyak_aarch64::T_m, src2);
        } else {
            mov(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src);
            fdiv(dst, pred / Xbyak_aarch64::T_m, src2);
        }
    }

    void uni_fsub(const Xbyak_aarch64::VReg4S &v1,
            const Xbyak_aarch64::VReg4S &v2, const Xbyak_aarch64::VReg4S &v3) {
        fsub(v1, v2, v3);
    }

    void uni_fsub(const Xbyak_aarch64::ZRegS &z1,
            const Xbyak_aarch64::ZRegS &z2, const Xbyak_aarch64::ZRegS &z3) {
        fsub(z1, z2, z3);
    }

    void uni_eor(const Xbyak_aarch64::VReg &v1, const Xbyak_aarch64::VReg &v2,
            const Xbyak_aarch64::VReg &v3) {
        eor(Xbyak_aarch64::VReg16B(v1.getIdx()),
                Xbyak_aarch64::VReg16B(v2.getIdx()),
                Xbyak_aarch64::VReg16B(v3.getIdx()));
    }

    void uni_eor(const Xbyak_aarch64::ZReg &z1, const Xbyak_aarch64::ZReg &z2,
            const Xbyak_aarch64::ZReg &z3) {
        eor(Xbyak_aarch64::ZRegD(z1.getIdx()),
                Xbyak_aarch64::ZRegD(z2.getIdx()),
                Xbyak_aarch64::ZRegD(z3.getIdx()));
    }

    void uni_ldr(
            const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::XReg &addr) {
        ldr(Xbyak_aarch64::QReg(dst.getIdx()), ptr(addr));
    }

    void uni_ldr(
            const Xbyak_aarch64::ZReg &dst, const Xbyak_aarch64::XReg &addr) {
        ldr(dst, ptr(addr));
    }

    void uni_str(
            const Xbyak_aarch64::VReg &src, const Xbyak_aarch64::XReg &addr) {
        str(Xbyak_aarch64::QReg(src.getIdx()), ptr(addr));
    }

    void uni_str(
            const Xbyak_aarch64::ZReg &src, const Xbyak_aarch64::XReg &addr) {
        str(src, ptr(addr));
    }

    /*
      Saturation facility functions. enable to prepare the register
      holding the saturation upperbound and apply the saturation on
      the floating point register
     */
    template <typename Vmm>
    void init_saturate_f32(Vmm vmm_lbound, Vmm vmm_ubound,
            Xbyak_aarch64::XReg reg_tmp, data_type_t idt, data_type_t odt) {
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;

        if (!((idt == f32) && utils::one_of(odt, u8, data_type::s8, s32)))
            return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data
        if (odt == u8) {
            if (isSVE) /* SVE is available. */
                dup(Xbyak_aarch64::ZRegS(vmm_lbound.getIdx()), 0);
            else if (mayiuse(asimd))
                movi(Xbyak_aarch64::VReg4S(vmm_lbound.getIdx()), 0);
            else
                assert(!"unreachable");
        }

        Xbyak_aarch64::ZRegS z_tmp(vmm_ubound.getIdx());
        Xbyak_aarch64::VReg4S v_tmp(vmm_ubound.getIdx());
        Xbyak_aarch64::WReg w_tmp(reg_tmp.getIdx());
        float saturation_ubound = types::max_value<float>(odt);
        mov_imm(w_tmp, float2int(saturation_ubound));
        if (isSVE) /* SVE is available. */
            dup(z_tmp, w_tmp);
        else
            dup(v_tmp, w_tmp);
    }

    template <typename Vmm>
    void saturate_f32(const Vmm &vmm, const Vmm &vmm_lbound,
            const Vmm &vmm_ubound, data_type_t odt,
            const Xbyak_aarch64::PReg &p_true) {
        // This function is used to saturate to odt in f32 before converting
        // to s32 in order to avoid bad saturation due to cvtps2dq
        // behavior (it returns INT_MIN if the f32 is out of the
        // s32 range)
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;

        if (!utils::one_of(odt, u8, data_type::s8, s32)) return;

        Xbyak_aarch64::VReg4S v_tmp(vmm.getIdx());
        Xbyak_aarch64::VReg4S v_lbound(vmm_lbound.getIdx());
        Xbyak_aarch64::VReg4S v_ubound(vmm_ubound.getIdx());
        Xbyak_aarch64::ZRegS z_tmp(vmm.getIdx());
        Xbyak_aarch64::ZRegS z_lbound(vmm_lbound.getIdx());
        Xbyak_aarch64::ZRegS z_ubound(vmm_ubound.getIdx());

        // no need to apply lower saturation bound when odt is
        // signed, as cvtps2dq will return MIN_INT if the value
        // does not fit
        if (odt == u8) {
            if (isSVE) /* SVE is available. */
                fmax(z_tmp, p_true / Xbyak_aarch64::T_m, z_lbound);
            else if (mayiuse(asimd))
                fmax(v_tmp, v_tmp, v_lbound);
            else
                assert(!"unreachable");
        }
        if (isSVE) /* SVE is available. */
            fmin(z_tmp, p_true / Xbyak_aarch64::T_m, z_ubound);
        else if (mayiuse(asimd))
            fmin(v_tmp, v_tmp, v_ubound);
        else
            assert(!"unreachable");
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true)
        : Xbyak_aarch64::CodeGenerator(code_size,
                (code_ptr == nullptr && use_autogrow) ? Xbyak_aarch64::AutoGrow
                                                      : code_ptr) {}
    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const uint8_t *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const uint8_t *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? status::success : status::runtime_error;
    }

private:
    const uint8_t *getCode() {
        this->ready();
        if (!is_initialized()) return nullptr;
        const uint8_t *code
                = reinterpret_cast<const uint8_t *>(CodeGenerator::getCode());
        register_jit_code(code, getSize() * CSIZE);
        return code;
    }

    static inline bool is_initialized() {
        /* At the moment, Xbyak_aarch64 does not have GetError()\
         so that return dummy result. */
        return true;
    }

protected:
    virtual void generate() = 0;
    const uint8_t *jit_ker_ = nullptr;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
