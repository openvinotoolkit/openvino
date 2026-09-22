// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <xbyak/xbyak.h>

#include <array>
#include <tuple>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "jit_kernel_emit.hpp"
#include "jit_kernel_ir.hpp"
#include "jit_kernel_target.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::intel_cpu {

// Instruction tags for the "instructions as data" dispatch.
// Each enum value maps to a single xbyak instruction in jit_kernel::lower().
// Adding a new instruction: add enum value here + one case in lower().
enum class Insn2 : uint8_t { vaddps, vsubps, vmulps, vmaxps, vminps, COUNT };
enum class Insn3 : uint8_t { fmadd231ps, fnmadd231ps, fmsub231ps, COUNT };

struct jit_kernel;

namespace internal {

template <size_t S>
struct reg_traits_by_size;
template <typename T>
struct reg_traits;
template <typename T, size_t N>
struct reg_traits<T[N]>;
template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct isa_traits;

template <>
struct reg_traits_by_size<1> {
    using type = Xbyak::Reg8;
    constexpr static size_t size = 1;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
};

template <>
struct reg_traits_by_size<2> {
    using type = Xbyak::Reg16;
    constexpr static size_t size = 2;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
};

template <>
struct reg_traits_by_size<4> {
    using type = Xbyak::Reg32;
    constexpr static size_t size = 4;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
};

template <>
struct reg_traits_by_size<8> {
    using type = Xbyak::Reg64;
    constexpr static size_t size = 8;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
};

template <>
struct reg_traits_by_size<16> {
    using type = Xbyak::Xmm;
    constexpr static size_t size = 16;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::sse41;
};

template <>
struct reg_traits_by_size<32> {
    using type = Xbyak::Ymm;
    constexpr static size_t size = 32;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::avx2;
};

template <>
struct reg_traits_by_size<64> {
    using type = Xbyak::Zmm;
    constexpr static size_t size = 64;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::avx512_core;
};

template <typename T>
struct reg_traits : public reg_traits_by_size<sizeof(T)> {};

template <size_t N>
struct vec_min_size {
    constexpr static size_t size = []() constexpr -> size_t {
        if (N <= 16) {
            return 16;
        }
        if (N <= 32) {
            return 32;
        }
        return 64;
    }();
};

template <typename T, size_t N>
struct reg_traits<T[N]> : public reg_traits_by_size<vec_min_size<sizeof(T[N])>::size> {};

template <>
struct reg_traits<float> {
    using type = Xbyak::Fpu;
    constexpr static size_t size = 10;  // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef;
};
template <>
struct reg_traits<double> : public reg_traits<float> {};

template <>
struct isa_traits<dnnl::impl::cpu::x64::cpu_isa_t::sse41> {
    struct reg {
        using type = Xbyak::Xmm;
        constexpr static size_t size = 4 * 4;  // in bytes
        constexpr static size_t length = 4;    // in dwords
    };
};

template <>
struct isa_traits<dnnl::impl::cpu::x64::cpu_isa_t::avx2> {
    struct reg {
        using type = Xbyak::Ymm;
        constexpr static size_t size = 8 * 4;  // in bytes
        constexpr static size_t length = 8;    // in dwords
    };
};

template <>
struct isa_traits<dnnl::impl::cpu::x64::cpu_isa_t::avx512_core> {
    struct reg {
        using type = Xbyak::Zmm;
        constexpr static size_t size = 16 * 4;  // in bytes
        constexpr static size_t length = 16;    // in dwords
    };
};

template <typename T, typename Tag>
class variable;
template <typename T>
class if_expression;
template <typename T>
class then_expression;
template <typename Reg>
using shared_reg = std::shared_ptr<Reg>;

template <typename Reg>
shared_reg<Reg> make_shared(Reg& reg, jit_kernel& kernel);

template <typename T>
class boolean_expression {
public:
    using reg_type = const typename reg_traits<T>::type;

    enum class type : uint8_t {
        eq,   // ==
        neq,  // !=
        ls,   // <
        gt,   // >
        le,   // <=
        ge    // >=
    };

    boolean_expression(jit_kernel& kernel, type t, const shared_reg<reg_type>& lhs, const shared_reg<reg_type>& rhs);
    boolean_expression(jit_kernel& kernel, type t, const shared_reg<reg_type>& lhs, T rhs);

private:
    void cmp(const Xbyak::Label& exit) const;

    jit_kernel& _kernel;
    type _type;
    shared_reg<reg_type> _lhs;
    shared_reg<reg_type> _rhs;
    T _rvalue;

    friend class if_expression<T>;
    friend class then_expression<T>;
};

template <typename T>
class then_expression {
public:
    explicit then_expression(if_expression<T>& expr);

    template <typename F>
    void _else(F&& fn);

private:
    if_expression<T>& _if_expr;
};

template <typename T>
class if_expression {
public:
    explicit if_expression(const boolean_expression<T>& expr) : _expr(expr) {}

    ~if_expression() {
        if (!_is_exit_valid) {
            _expr._kernel.assignL(_exit, _else);
        }
    }

    template <typename F>
    then_expression<T> _then(F&& fn) {
        using namespace Xbyak;

        _expr.cmp(_else);
        std::forward<F>(fn)();
        _expr._kernel.jmp(_exit, Xbyak::CodeGenerator::T_NEAR);
        _expr._kernel.L(_else);

        return then_expression<T>(*this);
    }

private:
    const boolean_expression<T>& _expr;
    Xbyak::Label _exit;
    Xbyak::Label _else;
    bool _is_exit_valid = false;

    friend class then_expression<T>;
};

using register_tag = struct register_tag {};
using memory_tag = struct memory_tag {};

template <typename T, typename Tag>
class variable_base;

template <typename T>
class variable_base<T, register_tag> {
public:
    using reg_type = const typename reg_traits<T>::type;

    variable_base& operator=(const variable_base&) = delete;

    variable_base(const variable_base& rhs);
    variable_base(variable_base&& rhs) noexcept;

    [[nodiscard]] reg_type& reg() const {
        return *_reg;
    }

    [[nodiscard]] const shared_reg<reg_type>& shreg() const {
        return _reg;
    }

    [[nodiscard]] jit_kernel_ir::value_id vid() const noexcept {
        return _vid;
    }

    // XByak relies on implicit conversions
    operator reg_type&() const {  // NOLINT(google-explicit-constructor)
        return reg();
    }

    operator Xbyak::RegExp() const {  // NOLINT(google-explicit-constructor)
        return reg();
    }

protected:
    variable_base(jit_kernel& krnl, const shared_reg<reg_type>& reg);
    variable_base(jit_kernel& krnl, jit_kernel_ir::value_id vid);
    ~variable_base() = default;

    jit_kernel& _kernel;
    // mutable so const-qualified move-assignment can rebind the underlying
    // register. `variable<>` is modeled as a const handle to a mutable
    // register slot: existing `operator=(const variable&)` already writes
    // into `*_reg` through `const` methods; move-assignment extends this by
    // letting the handle point to a different slot without copying contents.
    mutable shared_reg<reg_type> _reg;
    mutable jit_kernel_ir::value_id _vid = jit_kernel_ir::invalid_value;
};

template <typename T>
class variable_base<T, memory_tag> {
public:
    using reg_type = const typename reg_traits<T*>::type;

    variable_base& operator=(const variable_base&) = delete;

    variable_base(const variable_base& rhs);
    variable_base(variable_base&& rhs) noexcept;

    reg_type& reg() const {
        return *_addr;
    }

protected:
    variable_base(jit_kernel& krnl, const shared_reg<reg_type>& addr);
    ~variable_base() = default;

    jit_kernel& _kernel;
    shared_reg<const reg_type> _addr;
};

template <typename T>
class variable<T, register_tag>
    : public variable_base<std::enable_if_t<!std::is_floating_point_v<T>, T>, register_tag> {
public:
    using type = T;
    using base = variable_base<type, register_tag>;
    using reg_type = const typename base::reg_type;
    using arithmetic_type = std::conditional_t<std::is_pointer_v<T>, size_t, T>;

    variable(variable&&) noexcept = default;
    explicit variable(jit_kernel& krnl);
    variable(jit_kernel& krnl, const shared_reg<reg_type>& reg);
    variable(jit_kernel& krnl, jit_kernel_ir::value_id vid) : base(krnl, vid) {}

    std::conditional_t<std::is_pointer_v<T> && !std::is_pointer_v<std::remove_pointer_t<T>>,
                       variable<std::remove_pointer_t<T>, memory_tag>,
                       void>
    operator*() const {
        return variable<std::remove_pointer_t<T>, memory_tag>(base::_kernel, base::shreg());
    }

    // NOLINTBEGIN(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)
    const variable& operator=(reg_type& rhs) const {
        base::_kernel.mov(base::reg(), rhs);
        return *this;
    }
    template <typename U>
    const variable& operator=(U* rhs) const {
        // interpret pointers as size_t
        base::_kernel.mov(base::reg(), reinterpret_cast<size_t>(rhs));
        return *this;
    }
    const variable& operator=(arithmetic_type rhs) const {
        base::_kernel.mov(base::reg(), static_cast<size_t>(rhs));
        return *this;
    }

    // Copy-assign: emits a real `mov` to duplicate rhs's pointer value into
    // this handle's register. Use when you need a cursor that tracks the
    // source register independently (e.g. local pointer advanced by += over
    // a destination buffer). Mirrors variable<T[N]>::operator=(const variable&)
    // which emits vmovups for the same purpose.
    // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)
    const variable& operator=(const variable& rhs) const {
        base::_kernel.mov(base::reg(), rhs.reg());
        return *this;
    }

    // Move-assign: rebinds this handle to rhs's register slot — zero
    // instructions emitted. Mirrors variable<T[N]>::operator=(variable&&)
    // and extends the "value-transforming operations are by value" contract
    // (jit_kernel.md) to scalar pointer/integer variables so patterns like
    // `p = some_fn_returning_pointer_var()` stay free.
    // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)
    const variable& operator=(variable&& rhs) const noexcept {
        if (this != &rhs) {
            base::_reg = std::move(rhs._reg);
            base::_vid = rhs._vid;
        }
        return *this;
    }

    const variable& operator+=(reg_type& rhs) const {
        base::_kernel.add(base::reg(), rhs);
        return *this;
    }
    variable operator+(reg_type& rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res += rhs;
        return res;
    }
    const variable& operator+=(arithmetic_type rhs) const {
        base::_kernel.add(base::reg(), rhs);
        return *this;
    }
    variable operator+(arithmetic_type rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res += rhs;
        return res;
    }
    const variable& operator-=(reg_type& rhs) const {
        base::_kernel.sub(base::reg(), rhs);
        return *this;
    }
    variable operator-(reg_type& rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res -= rhs;
        return res;
    }
    const variable& operator-=(arithmetic_type rhs) const {
        base::_kernel.sub(base::reg(), rhs);
        return *this;
    }
    variable operator-(arithmetic_type rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res -= rhs;
        return res;
    }
    const variable& operator*=(reg_type& rhs) const {
        base::_kernel.imul(base::reg(), rhs);
        return *this;
    }
    variable operator*(reg_type& rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res *= rhs;
        return res;
    }
    const variable& operator*=(arithmetic_type rhs) const {
        base::_kernel.imul(base::reg(), base::reg(), static_cast<int>(rhs));
        return *this;
    }
    variable operator*(arithmetic_type rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res *= rhs;
        return res;
    }
    const variable& operator&=(reg_type& rhs) const {
        base::_kernel.and_(base::reg(), rhs);
        return *this;
    }
    variable operator&(reg_type& rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res &= rhs;
        return res;
    }
    const variable& operator&=(T rhs) const {
        base::_kernel.and_(base::reg(), rhs);
        return *this;
    }
    variable operator&(T rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res &= rhs;
        return res;
    }
    const variable& operator|=(reg_type& rhs) const {
        base::_kernel.or_(base::reg(), rhs);
        return *this;
    }
    variable operator|(reg_type& rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res |= rhs;
        return res;
    }
    const variable& operator|=(T rhs) const {
        base::_kernel.or_(base::reg(), rhs);
        return *this;
    }
    variable operator|(T rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res |= rhs;
        return res;
    }
    const variable& operator>>=(size_t rhs) const {
        base::_kernel.shr(base::reg(), rhs);
        return *this;
    }
    variable operator>>(size_t rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res >>= rhs;
        return res;
    }
    const variable& operator<<=(size_t rhs) const {
        base::_kernel.shl(base::reg(), rhs);
        return *this;
    }
    variable operator<<(size_t rhs) const {
        variable res(base::_kernel);
        res = base::reg();
        res <<= rhs;
        return res;
    }

    boolean_expression<T> operator==(const variable& rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::eq, base::shreg(), rhs.shreg());
    }

    boolean_expression<T> operator==(T rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::eq, base::shreg(), rhs);
    }

    boolean_expression<T> operator!=(const variable& rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::neq, base::shreg(), rhs.shreg());
    }

    boolean_expression<T> operator!=(T rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::neq, base::shreg(), rhs);
    }

    boolean_expression<T> operator<(const variable& rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::ls, base::shreg(), rhs.shreg());
    }

    boolean_expression<T> operator<(T rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::ls, base::shreg(), rhs);
    }

    boolean_expression<T> operator>(const variable& rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::gt, base::shreg(), rhs.shreg());
    }

    boolean_expression<T> operator>(T rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::gt, base::shreg(), rhs);
    }

    boolean_expression<T> operator<=(const variable& rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::le, base::shreg(), rhs.shreg());
    }

    boolean_expression<T> operator<=(T rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::le, base::shreg(), rhs);
    }

    boolean_expression<T> operator>=(const variable& rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::ge, base::shreg(), rhs.shreg());
    }

    boolean_expression<T> operator>=(T rhs) const {
        return boolean_expression<T>(base::_kernel, boolean_expression<T>::type::ge, base::shreg(), rhs);
    }

    // TODO: add necessary operations
};

template <typename T>
class variable<T, memory_tag> : public variable_base<T*, memory_tag> {
public:
    using type = T;
    using base = variable_base<type*, memory_tag>;
    using reg_type = const typename base::reg_type;

    variable(variable&&) noexcept = default;
    variable(jit_kernel& krnl, const shared_reg<reg_type>& reg);

    const variable& operator=(const variable<T, register_tag>& rhs) const;
};

template <typename T, size_t N>
class variable<T[N], register_tag> : public variable_base<T[N], register_tag> {
public:
    using type = T[N];
    using base = variable_base<type, register_tag>;
    using reg_type = const typename base::reg_type;
    constexpr static size_t length = N;

    variable(variable&&) noexcept = default;
    explicit variable(jit_kernel& krnl);
    variable(jit_kernel& krnl, const shared_reg<reg_type>& reg);
    variable(jit_kernel& krnl, jit_kernel_ir::value_id vid);

    const variable& operator=(reg_type& rhs) const {
        base::_kernel.uni_vmovups(base::reg(), rhs);
        return *this;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)
    const variable& operator=(const variable& rhs) const {
        base::_kernel.uni_vmovups(base::reg(), rhs.reg());
        return *this;
    }

    // Move-assignment: rebinds this handle to the rhs's register slot instead
    // of emitting a vmovups. The caller's previous register drops its
    // shared_reg reference (returned to the pool when refcount hits zero).
    // This is what makes `r = fma(a, b, c);` and similar by-value patterns
    // zero-cost compared to the old fluent in-place form — the vmovups that
    // would otherwise copy the temporary's contents into `r` is elided.
    // See "Value-transforming operations are by value" in jit_kernel.md.
    // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)
    const variable& operator=(variable&& rhs) const noexcept {
        if (this != &rhs) {
            base::_reg = std::move(rhs._reg);
            base::_vid = rhs._vid;
        }
        return *this;
    }

    const variable& operator=(const type& rhs) const {
        const type& cref = base::_kernel.constant(rhs);
        variable<const type*, register_tag> creg(base::_kernel);
        creg = &cref;
        base::_kernel.uni_vmovdqu(base::reg(), base::_kernel.ptr[creg]);
        return *this;
    }

    // By-value blend: emits 3-operand vblendps into a fresh destination
    // register. Same conventions as permute / shuffle / arithmetic ops —
    // see "Value-transforming operations are by value" in jit_kernel.md.
    variable blend(const variable& rhs, uint16_t mask) const {
        static_assert(std::is_same_v<T, float>, "vector blend requires float element type");
        using reg_type = typename reg_traits<type>::type;
        if constexpr (std::is_same_v<reg_type, Xbyak::Zmm>) {
            // AVX-512: vblendmps requires k-register. The mask constant
            // is an IR GPR value so the allocator assigns a non-conflicting
            // register. LLVM-style: every scratch is a virtual register.
            auto mask_vid = base::_kernel.ir_def_gpr({},
                [&k = base::_kernel, mask](const jit_kernel_ir::EmitContext& ctx) {
                    k.mov(Xbyak::Reg32(ctx.def->idx), mask);
                }, "blend_mask");
            return base::_kernel.template ir_def<N>(
                {base::vid(), rhs.vid(), mask_vid},
                [&k = base::_kernel](const jit_kernel_ir::EmitContext& ctx) {
                    k.kmovw(Xbyak::Opmask(1), Xbyak::Reg32(ctx.reads[2].idx));
                    k.vblendmps(reg_type(ctx.def->idx) | Xbyak::Opmask(1),
                                reg_type(ctx.reads[0].idx),
                                reg_type(ctx.reads[1].idx));
                }, "blend");
        } else {
            return base::_kernel.template ir_def<N>(
                {base::vid(), rhs.vid()},
                [&k = base::_kernel, mask](const jit_kernel_ir::EmitContext& ctx) {
                    k.uni_vblendps(reg_type(ctx.def->idx),
                                   reg_type(ctx.reads[0].idx),
                                   reg_type(ctx.reads[1].idx), mask);
                }, "blend");
        }
    }

    // Lane permutation (vpermps). Returns a fresh variable holding the
    // permuted contents; source is unchanged. By-value is the DSL convention
    // for all value-transforming operations — see "Value-transforming
    // operations are by value" in jit_kernel.md.
    variable permute(const std::array<uint8_t, N>& order) const {
        return permute(order.data());
    }

    variable permute(const uint8_t* order) const {
        static_assert(std::is_same_v<T, float>, "vector permute requires float element type");
        return base::_kernel.vec_permute(*this, order);
    }

    // In-lane shuffle with a 2-bit-per-lane imm8 selector (vshufps).
    // Returns a fresh variable. The imm8 is a runtime argument — `uni_vshufps`
    // bakes it into the emitted instruction at emit time, so there's no need
    // to make it a template parameter the way AOT intrinsic wrappers must.
    // Self-shuffle overload is the common form for lane-duplicating
    // deinterleave patterns.
    variable shuffle(uint8_t imm) const {
        return shuffle(*this, imm);
    }

    variable shuffle(const variable& other, uint8_t imm) const {
        static_assert(std::is_same_v<T, float>, "vector shuffle requires float element type");
        using reg_type = typename reg_traits<type>::type;
        return base::_kernel.template ir_def<N>(
            {base::vid(), other.vid()},
            [&k = base::_kernel, imm](const jit_kernel_ir::EmitContext& ctx) {
                k.uni_vshufps(reg_type(ctx.def->idx),
                              reg_type(ctx.reads[0].idx),
                              reg_type(ctx.reads[1].idx), imm);
            },
            "shufps");
    }

    // Elementwise clamp to [lo, hi]. Mirrors std::datapar::clamp — returns
    // by value, NaN behavior follows uni_vmaxps/uni_vminps. This is the one
    // intentional exception to the "one overload = one instruction" rule:
    // clamp lowers to vmaxps + vminps because that's the canonical SIMD
    // idiom on every supported ISA, and exposing the two halves separately
    // would just push the same pair onto every caller.
    variable clamp(const variable& lo, const variable& hi) const {
        static_assert(std::is_same_v<T, float>, "vector clamp requires float element type");
        auto clamped_lo = base::_kernel.vec_op(Insn2::vmaxps, *this, lo);
        return base::_kernel.vec_op(Insn2::vminps, clamped_lo, hi);
    }

    // Vector arithmetic. Dispatches through jit_kernel::vec_op() which handles
    // both eager mode (emit xbyak immediately) and IR mode (record into IR).
    // Float-only; integer vector ops and divide are intentionally not provided.
    template <typename U = T>
    variable operator+(const variable& rhs) const {
        static_assert(std::is_same_v<U, float>, "vector operator+ requires float element type");
        return base::_kernel.vec_op(Insn2::vaddps, *this, rhs);
    }

    template <typename U = T>
    variable operator-(const variable& rhs) const {
        static_assert(std::is_same_v<U, float>, "vector operator- requires float element type");
        return base::_kernel.vec_op(Insn2::vsubps, *this, rhs);
    }

    template <typename U = T>
    variable operator*(const variable& rhs) const {
        static_assert(std::is_same_v<U, float>, "vector operator* requires float element type");
        return base::_kernel.vec_op(Insn2::vmulps, *this, rhs);
    }
};

class stack_frame {
public:
    stack_frame(const stack_frame&) = delete;
    stack_frame& operator=(const stack_frame&) = delete;

    stack_frame(jit_kernel& kernel, size_t size, uint32_t alignment = 1);
    stack_frame(stack_frame&& rhs) noexcept;
    ~stack_frame();
    [[nodiscard]] const Xbyak::Reg64& pointer() const;
    void clear() const;

private:
    jit_kernel& _kernel;
    size_t _size;
    uint32_t _alignment;
};

template <typename T>
ov::element::Type type2precision();

dnnl::impl::cpu::x64::cpu_isa_t get_current_isa();

class consts_table {
public:
    consts_table(const consts_table&) = delete;
    consts_table& operator=(const consts_table&) = delete;

    consts_table() = default;
    const void* store(const void* data, size_t size);

private:
    static constexpr const size_t chunk_size = 512;
    using chunk = std::array<uint8_t, chunk_size>;
    std::list<chunk> _chunks;
    size_t _size{};
};

}  // namespace internal

struct jit_kernel : public dnnl::impl::cpu::x64::jit_generator_t, public arch_emitter {
    using reg_indices = std::vector<int>;
    template <typename T>
    using reg_traits = internal::reg_traits<T>;
    template <size_t S>
    using reg_traits_by_size = internal::reg_traits_by_size<S>;
    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    using isa_traits = internal::isa_traits<isa>;
    using stack_frame = internal::stack_frame;
    using register_tag = internal::register_tag;
    using memory_tag = internal::memory_tag;
    template <typename T, typename Tag = register_tag>
    using variable = internal::variable<T, Tag>;
    template <typename T>
    using if_expression = internal::if_expression<T>;
    template <typename T>
    using boolean_expression = internal::boolean_expression<T>;

public:
    // How many lanes of a vector operation are active — the DSL's
    // equivalent of LLVM's vector-predication operands, and deliberately
    // the same shape: a mask and an explicit vector length, either of
    // which may be absent.
    //
    //   all()                  — every lane; no predication needed
    //   elements(count)        — `count` leading lanes are active, count in
    //                            a GPR value. The portable form: every ISA
    //                            can express it (AVX-512 turns it into a
    //                            k-mask, SVE into whilelt, RVV into vsetvli,
    //                            AVX2 into vmaskmov or a scalar fallback).
    //   predicated(mask, count) — a materialized predicate value, with the
    //                            count kept alongside it for operations that
    //                            cannot consume a mask (interleaved stores
    //                            on x86, for instance).
    //
    // Nothing here names a register type: `mask` is an IR value in the Mask
    // register class, so the allocator tracks it like any other operand.
    // That is what LLVM does at IR level (`<n x i1>` mask + i32 evl) and at
    // MIR level (a predicate register operand).
    class vlen {
    public:
        // Every lane active. `terminal` is orthogonal to the active length:
        // a fully-active iteration can still be the last one, which is the
        // case for the final peeled step of a constant trip count.
        static vlen all(bool terminal = false) {
            vlen v;
            v._terminal = terminal;
            return v;
        }

        static vlen elements(jit_kernel_ir::value_id count, bool terminal = false) {
            vlen v;
            v._count = count;
            v._terminal = terminal;
            return v;
        }

        static vlen predicated(jit_kernel_ir::value_id mask,
                               jit_kernel_ir::value_id count = jit_kernel_ir::invalid_value,
                               bool terminal = false) {
            vlen v;
            v._mask = mask;
            v._count = count;
            v._terminal = terminal;
            return v;
        }

        // All lanes active, statically known. Lets a target pick the
        // cheaper unpredicated encoding.
        [[nodiscard]] bool is_all() const noexcept {
            return _count == jit_kernel_ir::invalid_value && _mask == jit_kernel_ir::invalid_value;
        }
        [[nodiscard]] bool has_mask() const noexcept {
            return _mask != jit_kernel_ir::invalid_value;
        }
        [[nodiscard]] bool has_count() const noexcept {
            return _count != jit_kernel_ir::invalid_value;
        }
        [[nodiscard]] jit_kernel_ir::value_id mask() const noexcept { return _mask; }
        [[nodiscard]] jit_kernel_ir::value_id count() const noexcept { return _count; }

        // Last iteration of the enclosing loop: nothing after it reads the
        // induction state, so pointer bumps can be skipped. True for an
        // epilogue tail, false inside any loop that iterates.
        [[nodiscard]] bool is_terminal() const noexcept { return _terminal; }

    private:
        jit_kernel_ir::value_id _mask = jit_kernel_ir::invalid_value;
        jit_kernel_ir::value_id _count = jit_kernel_ir::invalid_value;
        bool _terminal = false;
    };

    template <typename T, typename U>
    Xbyak::Address argPtr(U T::*member) const {
        auto memPtr = &(reinterpret_cast<const T*>(0)->*member);
        const size_t offs = reinterpret_cast<const char*>(memPtr) - reinterpret_cast<const char*>(0);
        return address_frame(sizeof(U))[param1 + offs];
    }

    // Load a kernel argument. In IR mode (LLVM-style): creates a GPR IR value
    // loaded from the params struct at lowering time. In eager mode: allocates
    // a physical register immediately.
    template <typename T, typename U>
    variable<U> arg(U T::*member) {
        if (_ir) {
            auto offs = member_offset(member);
            auto vid = _ir->def({}, [this, offs](const jit_kernel_ir::EmitContext& ctx) {
                auto dst = Xbyak::Reg64(ctx.def->idx);
                if (sizeof(U) < sizeof(size_t)) {
                    movzx(dst, address_frame(sizeof(U))[param1 + offs]);
                } else {
                    mov(dst, address_frame(sizeof(U))[param1 + offs]);
                }
            }, "arg", jit_kernel_ir::RegisterClass::GPR);
            return variable<U>(*this, vid);
        }
        using traits = internal::reg_traits<U>;
        using reg_type = typename traits::type;
        const auto& res = reserve<reg_type>();
        if (sizeof(T) < traits::size) {
            movzx(res, argPtr(member));
        } else {
            mov(res, argPtr(member));
        }
        return {*this, internal::make_shared(res, *this)};
    }

    template <typename CastU, typename T, typename U>
    variable<CastU> arg(U T::*member) {
        if (_ir) {
            auto offs = member_offset(member);
            auto vid = _ir->def({}, [this, offs](const jit_kernel_ir::EmitContext& ctx) {
                auto dst = Xbyak::Reg64(ctx.def->idx);
                if (sizeof(U) < sizeof(size_t)) {
                    movzx(dst, address_frame(sizeof(U))[param1 + offs]);
                } else {
                    mov(dst, address_frame(sizeof(U))[param1 + offs]);
                }
            }, "arg", jit_kernel_ir::RegisterClass::GPR);
            return variable<CastU>(*this, vid);
        }
        using traits = internal::reg_traits<U>;
        using reg_type = typename traits::type;
        const auto& res = reserve<reg_type>();
        if (sizeof(T) < traits::size) {
            movzx(res, argPtr(member));
        } else {
            mov(res, argPtr(member));
        }
        return {*this, internal::make_shared(res, *this)};
    }

private:
    template <typename T, typename U>
    static size_t member_offset(U T::*member) {
        auto memPtr = &(reinterpret_cast<const T*>(0)->*member);
        return reinterpret_cast<const char*>(memPtr) - reinterpret_cast<const char*>(0);
    }

public:

    explicit jit_kernel(const char* name);

    template <typename RegType>
    const RegType& reserve();

    template <typename RegType>
    void free(const RegType& reg);

    template <typename T>
    void copy(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size);
    template <typename T>
    void copy(const Xbyak::Address& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size);

    template <typename DstT, size_t N, typename SrcT>
    void load(const variable<DstT[N]>& dst, const variable<SrcT>& src, size_t length = N);
    template <typename DstT, size_t N, typename SrcT>
    void load(const variable<DstT[N]>& dst, const variable<SrcT>& src, const variable<size_t>& length);
    template <typename DstT, typename SrcT, size_t N>
    void store(const variable<DstT>& dst, const variable<SrcT[N]>& src, size_t length = N);
    template <typename DstT, typename SrcT, size_t N>
    void store(const variable<DstT>& dst, const variable<SrcT[N]>& src, const variable<size_t>& length);

    // Fused multiply-add:   fma(a, b, c) ->  a*b + c       (one vfmadd231ps + seed move)
    // Fused negative MAdd:  fnma(a, b, c) -> c - a*b       (one vfnmadd231ps + seed move)
    //
    // Both return by value. The seed move (vmovups res, c) is required because
    // vfmadd231ps writes into its destination with semantics "dst = dst + src1*src2" —
    // we need `dst` to already hold `c`. On modern x86 the seed move is elided by
    // register renaming (zero ALU uops at runtime).
    //
    // Exposed as explicit DSL primitives because xbyak emits what we tell it to emit —
    // there is no compiler-level contraction in the JIT path, unlike compiled C++.
    // See also std::datapar::fma; fnma has no std counterpart (known gap in the spec).
    template <size_t N>
    variable<float[N]> fma(const variable<float[N]>& a,
                           const variable<float[N]>& b,
                           const variable<float[N]>& c);
    template <size_t N>
    variable<float[N]> fmsub(const variable<float[N]>& a,
                             const variable<float[N]>& b,
                             const variable<float[N]>& c);
    template <size_t N>
    variable<float[N]> fnma(const variable<float[N]>& a,
                            const variable<float[N]>& b,
                            const variable<float[N]>& c);

    // 3-way interleaved store: writes three planar streams to memory in
    // interleaved order (a0 b0 c0 a1 b1 c1 ...). Mirrors ARM VST3 / SVE ST3 /
    // RVV vsseg3 semantics — on those ISAs a port can emit a single store
    // instruction. On x86 this lowers to an internal reg-only interleave
    // (3 × vpermps + 6 × vblendps) followed by three plain stores.
    //
    // Convention matches Highway's StoreInterleaved3 and std::datapar's
    // simd_unchecked_store (P1928R15 §29.10.7.6): `void` return, destination
    // is read-only from the function's point of view, and the caller walks
    // its own cursor externally with `dst += 3 * N * sizeof(T)` after the
    // call. This keeps the store focused on memory writes and leaves cursor
    // arithmetic to the caller, which composes naturally with unrolled loops
    // and loop prologues/epilogues.
    //
    // The `count` overload handles the vectorized tail: it emits the full
    // interleave to a stack slot and then copies `count * 3` elements out.
    // This is the portable fallback that every non-RVV backend can reuse
    // (no ISA has a variable-length masked interleaved store).
    //
    // API shape follows ISA shape: 3/4-way interleave is memory-coupled in
    // hardware, so there is no reg-only `interleave3` primitive. Callers that
    // want interleaved RGB bytes in memory should call store_interleaved3
    // directly rather than materializing the layout in registers first.
    //
    // NOTE: The full-width overload currently pays one emitted `mov` for a
    // local cursor (`auto p = var<T*>(); p = dst; ...`) because the underlying
    // jit_store_emitter takes a bare register for the destination and has no
    // displacement-addressing path. Adding `ptr[dst + offset]` support to the
    // emitter would let this overload emit three displaced stores with zero
    // cursor arithmetic — that's a DSL-level fix that benefits every caller,
    // not a store_interleaved3-specific concern.
    template <typename T, size_t N>
    void store_interleaved3(const variable<T*>& dst,
                            const variable<float[N]>& a,
                            const variable<float[N]>& b,
                            const variable<float[N]>& c);

    template <typename T, size_t N>
    void store_interleaved3(const variable<T*>& dst,
                            const variable<float[N]>& a,
                            const variable<float[N]>& b,
                            const variable<float[N]>& c,
                            const variable<size_t>& count);

    // Same, with the three stores predicated instead of bounced through a
    // stack slot. Selected by supports_masked_interleaved_access().
    template <typename T, size_t N>
    void store_interleaved3_predicated(const variable<T*>& dst,
                                       const variable<float[N]>& a,
                                       const variable<float[N]>& b,
                                       const variable<float[N]>& c,
                                       const variable<size_t>& count);

    // 2-way deinterleave: separates even/odd elements across two vectors.
    //   in:  a = [x0 x1 x2 x3 ...], b = [xN xN+1 xN+2 xN+3 ...]
    //   out: evens = [x0 x2 x4 ...], odds = [x1 x3 x5 ...]
    //
    // Portable semantic op. Lowering:
    //   ARM NEON/SVE: UZP1 / UZP2 (single instruction each)
    //   AVX2:         vperm2i128 + vshufps
    //   AVX-512:      vshuff32x4 + vshufps
    template <size_t N>
    std::pair<variable<float[N]>, variable<float[N]>> deinterleave2(
        const variable<float[N]>& a,
        const variable<float[N]>& b);

    // 2-way interleave: merges even/odd streams back into sequential order.
    //   in:  evens = [e0 e1 e2 ...], odds = [o0 o1 o2 ...]
    //   out: lo = [e0 o0 e1 o1 ...], hi = [eN/2 oN/2 eN/2+1 oN/2+1 ...]
    //
    // Portable semantic op. Lowering:
    //   ARM NEON/SVE: ZIP1 / ZIP2 (single instruction each)
    //   AVX2:         vunpcklps + vunpckhps + vperm2i128
    //   AVX-512:      vpermq + vunpcklps + vunpckhps
    template <size_t N>
    std::pair<variable<float[N]>, variable<float[N]>> interleave2(
        const variable<float[N]>& evens,
        const variable<float[N]>& odds);

private:
    // Internal reg-only 3-way interleave helper. Building block for
    // store_interleaved3's x86 lowering; not part of the public DSL because
    // holding interleaved RGB lanes in registers has no downstream use
    // (the lanes are heterogeneous — no SIMD math applies) and NEON/SVE/RVV
    // cannot materialize this form efficiently anyway.
    //
    //   in0 = a0 a1 a2 a3 ... aN-1
    //   in1 = b0 b1 b2 b3 ... bN-1
    //   in2 = c0 c1 c2 c3 ... cN-1
    //
    //   out0, out1, out2 hold a0 b0 c0 a1 b1 c1 ... aN-1 bN-1 cN-1
    //   packed into N lanes per vector.
    template <size_t N>
    std::tuple<variable<float[N]>, variable<float[N]>, variable<float[N]>> interleave_regs(
        const variable<float[N]>& a,
        const variable<float[N]>& b,
        const variable<float[N]>& c);

public:

    template <typename B, typename E, typename S = size_t>
    void foreach (const B& begin,
                  const E& end,
                  const std::function<void(const variable<size_t>&)>& fn,
                  const S& step = 1);

    // Predicated foreach: single loop processing all elements including
    // the tail. The body receives a k-register mask (AVX-512) that is
    // all-ones for full iterations and partial for the last iteration.
    // Use ir_load_masked / ir_store_masked inside the body.
    //
    // On AVX-512: one loop, mask computed per iteration, zero overhead
    //             for full iterations.
    // On AVX2:    main loop (unmasked, body called with no_mask) +
    //             tail (body called with partial mask outside the loop).
    // unroll=1: single body per loop iteration (default).
    // unroll>1: body recorded N times per iteration, each with its own
    //           mask setup. Loop iterates ceil(total/(N*unroll)) times.
    //           Equivalent to legacy manual unrolling.
    template <size_t N>
    void foreach_predicated(const variable<size_t>& total_count,
                            const std::function<void(const vlen&)>& fn,
                            size_t unroll = 1);

    template <typename T>
    variable<T> var();
    template <typename T>
    variable<T> var(const T& val);
    // Construct-and-load in one step: reserve a fresh vector register and
    // emit the type-converting load from `src`. Full-vector, compile-time and
    // runtime-length overloads mirror the existing `load(dst, src, [length])`
    // methods.
    template <typename T, typename SrcT>
    variable<T> var(const variable<SrcT>& src);
    template <typename T, typename SrcT>
    variable<T> var(const variable<SrcT>& src, size_t length);
    template <typename T, typename SrcT>
    variable<T> var(const variable<SrcT>& src, const variable<size_t>& length);

    template <typename T>
    const T& constant(const T& c);
    template <typename T>
    const T* constant(const T* c, size_t size);

    stack_frame stack(size_t size, uint32_t alignment = 1);

    template <typename T>
    if_expression<T> _if(const boolean_expression<T>& expr) const;

    // IR-mode conditional: records then/else bodies as nested regions.
    // Condition must be set via ir_cmp() before calling this.
    // `skip_on` is the condition that skips the THEN body.
    template <typename ThenFn, typename ElseFn>
    void ir_if(cond skip_on, ThenFn&& then_fn, ElseFn&& else_fn);

    // Overload without else branch.
    template <typename ThenFn>
    void ir_if(cond skip_on, ThenFn&& then_fn);

    // ── arch_emitter: the x86 realization ─────────────────────────────
    // One instruction (or one short expansion) per portable operation.
    // Nothing here decides anything — the IR shape of each operation is
    // fixed by the portable caller; see jit_kernel_emit.hpp.

    [[nodiscard]] jit_kernel_ir::EmitFn gpr_copy() const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_set(std::uint64_t imm) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_add_imm(std::uint64_t imm) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_shr_imm(unsigned shift) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_and_imm(std::uint64_t imm) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_imul_imm(std::uint64_t imm) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_bump(std::int64_t delta) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_offset(std::size_t imm) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_cmp_imm(std::uint64_t imm) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn gpr_cmp_reg() const override;
    [[nodiscard]] jit_kernel_ir::EmitFn clamped_len(std::size_t lanes) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn lane_mask_bits(std::size_t lanes) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn mask_from_bits(std::size_t lanes) const override;
    [[nodiscard]] label_ref make_label() const override;
    [[nodiscard]] jit_kernel_ir::EmitFn place_label(const label_ref& at) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn branch(cond on, const label_ref& to) const override;
    [[nodiscard]] jit_kernel_ir::EmitFn branch_always(const label_ref& to) const override;

    void uni_vpermps(const Xbyak::Xmm& x1, const uint8_t mask[4], const Xbyak::Operand& op);
    void uni_vpermps(const Xbyak::Ymm& y1, const uint8_t mask[8], const Xbyak::Operand& op);
    void uni_vpermps(const Xbyak::Zmm& z1, const uint8_t mask[16], const Xbyak::Operand& op);
    void uni_vblendps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, uint16_t mask);
    void uni_vblendps(const Xbyak::Ymm& y1, const Xbyak::Ymm& y2, uint16_t mask);
    void uni_vblendps(const Xbyak::Zmm& z1, const Xbyak::Zmm& z2, uint16_t mask);
    void uni_vblendps(const Xbyak::Xmm& dst, const Xbyak::Xmm& src1, const Xbyak::Xmm& src2, uint16_t mask);
    void uni_vblendps(const Xbyak::Ymm& dst, const Xbyak::Ymm& src1, const Xbyak::Ymm& src2, uint16_t mask);
    void uni_vblendps(const Xbyak::Zmm& dst, const Xbyak::Zmm& src1, const Xbyak::Zmm& src2, uint16_t mask);

    // ── IR mode ──────────────────────────────────────────────────────────
    // Opt-in recording mode. DSL calls record into IR instead of emitting
    // xbyak immediately. end_ir() runs allocation + lowering.

    // Element type behind a pointer type — spelled once so the memory
    // operations can ask the target about it.
    template <typename PtrT>
    using elem_of = std::remove_cv_t<std::remove_pointer_t<PtrT>>;

    // Capabilities of the ISA this kernel is generated for. Queried before
    // recording; see jit_kernel_target.hpp.
    // The target this kernel is generating for. Defaults to the host, and
    // is injectable so that decisions taken for a capability the host does
    // not have can be tested — the branches for SVE's predicated
    // interleaved store and RVV's register-only addressing are otherwise
    // unreachable, hence unverified. LLVM tests the same way: a target
    // triple and subtarget features are inputs, not properties of the
    // machine running the compiler.
    [[nodiscard]] const vector_target& target() const {
        return _target != nullptr ? *_target : host_vector_target();
    }

    void set_target(const vector_target& t) { _target = &t; }

    // IR mode is always active between begin_ir() and end_ir().
    [[nodiscard]] bool ir_mode() const noexcept { return _ir != nullptr; }
    void begin_ir();
    void end_ir();

    // Declare the vector width this kernel works in, in bits. A kernel that
    // is exclusively 512-bit gets the full AVX-512 register file (32
    // registers) in IR mode; anything else is limited to the low 16, because
    // xmm16..31 / ymm16..31 are EVEX-only and VEX-encoded instructions
    // cannot address them. Call before end_ir().
    void set_vec_width(size_t bits) noexcept { _vec_width_bits = bits; }

    // Generic dispatch: binary vector op. Handles IR/eager branching,
    // register allocation, and width dispatch in one place.
    template <size_t N>
    variable<float[N]> vec_op(Insn2 insn,
                              const variable<float[N]>& a,
                              const variable<float[N]>& b);

    // ── Accumulation: update a value in place ─────────────────────────
    //
    //   acc op= x          acc ±= a * b
    //
    // Unlike the expression forms above, these do not produce a new value:
    // they redefine `acc`, so the update survives a loop back edge. That
    // is the difference between a running sum and a body that recomputes
    // from the initial value every trip, and the latter is what writing
    // `acc = acc + x` inside a loop actually records — a fresh value the
    // next iteration never sees.
    //
    // The accumulator must be defined before the loop it is updated in.
    // Nothing checks that: the verifier has no dominance check, and
    // liveness over-approximates rather than rejecting, so an accumulator
    // first defined inside the loop reads whatever the register held.
    //
    // LLVM would write a PHI at the loop header; after PHIElimination and
    // TwoAddressInstructionPass it becomes exactly this — one register,
    // initialized in the preheader, redefined in place in the body.
    template <size_t N>
    void ir_accumulate(const variable<float[N]>& acc, Insn2 insn,
                       const variable<float[N]>& x);

    template <size_t N>
    void ir_accumulate(const variable<float[N]>& acc, Insn3 insn,
                       const variable<float[N]>& a,
                       const variable<float[N]>& b);

    // Copy a vector value. In IR mode, records a copy op (coalescing hint).
    // In eager mode, emits vmovups into a fresh register.
    template <size_t N>
    variable<float[N]> vec_copy(const variable<float[N]>& src);

    // Permute vector lanes. ISA dispatch: SSE uses shufps (imm8),
    // AVX2/AVX-512 express the permute table as an IR-managed value
    // so the allocator tracks the extra register.
    template <size_t N>
    variable<float[N]> vec_permute(const variable<float[N]>& src,
                                   const uint8_t* order);

    // Type-converting load/store. The pointer must be an IR value (call
    // arg() inside IR mode). The element type of the pointer selects the
    // conversion: f32 (plain move), u8, f16, bf16.
    //
    // `vl` says how many lanes are active: vlen::all() for a whole
    // vector, vlen::masked(k) for AVX-512 masked access, vlen::count(v)
    // for a runtime element count.
    template <size_t N, typename PtrT>
    variable<float[N]> ir_load(const variable<PtrT>& ptr, size_t byte_offset = 0,
                               const vlen& vl = vlen::all());

    template <size_t N, typename PtrT>
    variable<float[N]> ir_load(const variable<PtrT>& ptr, const vlen& vl) {
        return ir_load<N>(ptr, size_t{0}, vl);
    }

    template <size_t N, typename PtrT, typename ElemT>
    void ir_store(const variable<PtrT>& ptr, size_t byte_offset,
                  const variable<ElemT[N]>& val, const vlen& vl = vlen::all());

    // Convenience: ir_store without offset.
    template <size_t N, typename PtrT, typename ElemT>
    void ir_store(const variable<PtrT>& ptr, const variable<ElemT[N]>& val,
                  const vlen& vl = vlen::all()) {
        ir_store<N>(ptr, size_t{0}, val, vl);
    }

    // Low-level bridge helpers for IR mode: record a def/use op with a
    // custom emit closure. Used by ir_load/ir_store internally and
    // available for custom bridging when needed.
    template <size_t N>
    variable<float[N]> ir_def(std::vector<jit_kernel_ir::value_id> reads,
                              jit_kernel_ir::EmitFn emit,
                              const char* name = "");
    void ir_use(std::vector<jit_kernel_ir::value_id> reads,
                jit_kernel_ir::EmitFn emit,
                const char* name = "");

    // Define a GPR IR value. The allocator assigns a physical GPR register.
    // At lowering time, ctx.def->idx is the physical Reg64 index.
    jit_kernel_ir::value_id ir_def_gpr(std::vector<jit_kernel_ir::value_id> reads,
                                        jit_kernel_ir::EmitFn emit,
                                        const char* name = "");

    // Define a predicate (Mask class) IR value. ctx.def->idx is the
    // physical mask register index at lowering time.
    jit_kernel_ir::value_id ir_def_mask(std::vector<jit_kernel_ir::value_id> reads,
                                        jit_kernel_ir::EmitFn emit,
                                        const char* name = "");

    // Materialize a predicate for the first `count` lanes of an N-lane
    // vector — the DSL's llvm.get.active.lane.mask. Two ops, because that
    // is how the hardware works: compute the lane bits in a GPR, then move
    // them into a predicate register.
    template <size_t N>
    jit_kernel_ir::value_id ir_active_lane_mask(const variable<size_t>& count);

    // Just the lane bits in a GPR, without moving them into the predicate
    // file. Split out because one computation can feed several predicates:
    // a 3-way interleaved store of `count` elements wants the low 3*count
    // bits, sliced into three predicates by shifting.
    jit_kernel_ir::value_id ir_lane_mask_bits(jit_kernel_ir::value_id count_vid, size_t lanes);

    // Same, for an active length known at kernel-build time: the lane bits
    // are an immediate, so the runtime compare/bts/dec disappears and no
    // early-clobber constraint is needed (the def has no reads).
    template <size_t N>
    jit_kernel_ir::value_id ir_const_lane_mask(size_t active);

    // A compile-time constant in a GPR IR value.
    variable<size_t> ir_gpr_imm(size_t value);

    // `ptr + byte_offset` as an IR pointer value. Returns the original
    // value id when the offset is zero, so the common case costs nothing.
    template <typename PtrT>
    jit_kernel_ir::value_id ir_offset_ptr(const variable<PtrT>& base, size_t byte_offset);

    // GPR arithmetic helpers — IRBuilder-style wrappers over ir_def_gpr.
    // Each creates a fresh IR GPR value = src OP imm.
    variable<size_t> ir_shr(const variable<size_t>& src, int shift);
    variable<size_t> ir_and(const variable<size_t>& src, size_t mask);
    variable<size_t> ir_add(const variable<size_t>& src, size_t val);
    variable<size_t> ir_imul(const variable<size_t>& src, size_t val);

    // Runtime-count partial load. Loads `count` elements from `src_ptr`
    // into a float[N] vector, zeroing elements beyond `count`.
    // Safe for any count in [0, N] — no reads past the buffer.
    // Internally: clear stack slot → scalar copy → full-width vector load.
    template <size_t N, typename PtrT>
    variable<float[N]> ir_load_partial(const variable<PtrT>& src_ptr,
                                       const variable<size_t>& count,
                                       size_t byte_offset = 0);

    // Runtime-count partial store — the mirror of ir_load_partial. Writes
    // exactly `count` elements to `dst_ptr` and touches nothing past them.
    // Internally: full-width (type-converting) store into a stack slot →
    // scalar copy of `count` elements to the destination.
    // Both pointers must be IR values (call arg() inside IR mode).
    template <size_t N, typename PtrT, typename ElemT>
    void ir_store_partial(const variable<PtrT>& dst_ptr,
                          const variable<ElemT[N]>& val,
                          const variable<size_t>& count,
                          size_t byte_offset = 0);

    // Lowering: maps instruction enum to xbyak call. The only code that
    // names specific xbyak instructions. Templated on register type so
    // uni_* overloads resolve naturally from the caller's width.
    template <typename Reg>
    void lower(Insn2 insn, const Reg& d, const Reg& s1, const Reg& s2);

    // Memory forms — the rm counterparts of the register instructions
    // above. x86 allows one memory operand, always the last one, which is
    // why FoldMemoryOperandsPass only ever folds a single read.
    template <typename Reg>
    void lower(Insn2 insn, const Reg& d, const Reg& s1, const Xbyak::Address& s2);

    // Destructive ternary dispatch: result = seed ± a * b.
    // The emit closure handles the tied-operand constraint: vmovups(def, seed)
    // before the FMA. If the allocator assigns the same register for def
    // and seed, the vmovups is a self-move (zero cost on modern x86).
    template <size_t N>
    variable<float[N]> vec_op(Insn3 insn,
                              const variable<float[N]>& seed,
                              const variable<float[N]>& a,
                              const variable<float[N]>& b);

    // Destructive FMA: d = d ± s1 * s2. Seed copy handled by vec_copy().
    template <typename Reg>
    void lower(Insn3 insn, const Reg& d, const Reg& s1, const Reg& s2);

    template <typename Reg>
    void lower(Insn3 insn, const Reg& d, const Reg& s1, const Xbyak::Address& s2);

    // ── Syntactic sugar: instruction functors ──────────────────────────
    // Lightweight callable that binds an instruction enum to this kernel.
    // Deduces vector width N from the variable arguments.
    //   auto c = vaddps(a, b);   // instead of vec_op(Insn2::vaddps, a, b)
    struct Op2 {
        jit_kernel* self;
        Insn2 insn;
        template <size_t N>
        variable<float[N]> operator()(const variable<float[N]>& a,
                                      const variable<float[N]>& b) const {
            return self->vec_op(insn, a, b);
        }
    };

    struct Op3 {
        jit_kernel* self;
        Insn3 insn;
        template <size_t N>
        variable<float[N]> operator()(const variable<float[N]>& a,
                                      const variable<float[N]>& b,
                                      const variable<float[N]>& c) const {
            return self->vec_op(insn, a, b, c);
        }
    };

    Op2 vaddps{this, Insn2::vaddps};
    Op2 vsubps{this, Insn2::vsubps};
    Op2 vmulps{this, Insn2::vmulps};
    Op2 vmaxps{this, Insn2::vmaxps};
    Op2 vminps{this, Insn2::vminps};
    Op3 vfmadd231ps{this, Insn3::fmadd231ps};
    Op3 vfnmadd231ps{this, Insn3::fnmadd231ps};
    Op3 vfmsub231ps{this, Insn3::fmsub231ps};

    // Broadcast a scalar from memory into an IR vector variable.
    //
    // The Address overload is for addresses built from registers outside the
    // allocator's jurisdiction (the constants-table register, for example).
    // Prefer the pointer overload when the base is a kernel argument: it
    // takes the pointer as an IR value, so the allocator sees the
    // dependency.
    template <size_t N>
    variable<float[N]> ir_broadcast(const Xbyak::Address& addr);

    template <size_t N, typename PtrT>
    variable<float[N]> ir_broadcast(const variable<PtrT>& ptr, size_t byte_offset = 0);

    // Zero vector — vxorps into a fresh register.
    template <size_t N>
    variable<float[N]> ir_zero();

    // GPR compare — sets flags, no result. In IR mode records a deferred
    // cmp; in eager mode emits immediately. Operands are scalar variables
    // or immediates (not IR-managed vector values).
    template <typename A, typename B>
    void ir_cmp(const A& a, const B& b);

    void postamble();

    const Xbyak::AddressFrame& address_frame(size_t size) const;
    const reg_indices& free_x64regs() const;
    const reg_indices& free_rmmregs() const;

private:
    reg_indices _free_x64regs;
    reg_indices _free_rmmregs;
    internal::consts_table _consts;
    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> _emitters;

    // IR state — non-null between begin_ir() and end_ir().
    std::unique_ptr<jit_kernel_ir::IR> _ir;

    // Vector width declared via set_vec_width(); 0 = unspecified.
    size_t _vec_width_bits = 0;

    // Active-lane masks already materialized in the body being recorded,
    // keyed by the count they were derived from. Cleared per body instance
    // (per loop iteration, per unroll step, per epilogue phase), so several
    // accesses sharing one active length share one predicate instead of
    // recomputing it. Reset by lane_mask_scope.
    std::unordered_map<jit_kernel_ir::value_id, jit_kernel_ir::value_id> _lane_masks;

    // RAII: a body instance starts with no materialized predicates.
    struct lane_mask_scope {
        explicit lane_mask_scope(jit_kernel& k) : _kernel(k) { _kernel._lane_masks.clear(); }
        ~lane_mask_scope() { _kernel._lane_masks.clear(); }
        lane_mask_scope(const lane_mask_scope&) = delete;
        lane_mask_scope& operator=(const lane_mask_scope&) = delete;
        jit_kernel& _kernel;
    };

    // Stack allocations requested by ir_alloca(). Offsets computed in end_ir()
    // before lowering. One sub/add rsp pair for the total.
    struct AllocaRequest {
        size_t size;
        size_t alignment;
        size_t offset = 0;  // filled before lowering
    };
    std::vector<AllocaRequest> _alloca_requests;

    // See set_peel_limit(). Initialized from OV_JIT_IR_PEEL.
    size_t _peel_limit = default_peel_limit();

    // See set_target(). Null = the host target.
    const vector_target* _target = nullptr;

    static size_t default_peel_limit();

public:
    // Pointer with stride — carries the number of elements accessed per
    // vector iteration. foreach_with_epilogue uses the stride to compute
    // per-pointer partial counts and auto-advance.
    // Mirrors RVV's element group size — the stride IS the per-pointer vsetvli.
    template <typename T>
    struct ir_ptr {
        variable<T*> ptr;
        size_t stride;  // elements per iteration (N for full, N/2 for subsampled, etc.)

        // Constant byte displacement accumulated by ir_advance() in
        // straight-line code, added to every access off this pointer. The
        // address is a base register plus a displacement, which is what a
        // target's addressing mode is — LLVM carries the same two fields in
        // X86AddressMode{Base, Disp} and asks isLegalAddressingMode whether
        // a given Disp is free.
        //
        // Nonzero only where the target says the displacement rides in the
        // instruction (see vector_target::is_legal_access_offset); otherwise
        // ir_advance emits a real increment and this stays 0.
        size_t disp = 0;

        // The displacement in whole vector registers, which is the only
        // form SVE can encode ([x, #imm, MUL VL]).
        [[nodiscard]] size_t disp_vectors() const {
            const size_t vec_bytes = stride * sizeof(T);
            return vec_bytes != 0 ? disp / vec_bytes : 0;
        }

        jit_kernel_ir::value_id vid() const { return ptr.vid(); }
    };

    template <typename T>
    ir_ptr<T> make_ir_ptr(variable<T*>&& ptr, size_t stride) {
        return {std::move(ptr), stride};
    }

    // Vectorized loop over `count` elements. The body is called with the
    // active length for the iteration and passes it to its memory
    // operations; how the leftover elements are handled is the target's
    // decision, not the kernel author's:
    //
    //   mask     -> one predicated loop (AVX-512, SVE)
    //   epilogue -> full-width loop plus a counted tail (AVX2, SSE, NEON)
    //   length   -> one loop with a per-iteration vector length (RVV)
    //
    // Mirrors LLVM's TailFoldingStyle selection in the loop vectorizer:
    // the source shape is the same either way, and the target picks.
    template <size_t N>
    void foreach_vec(const variable<size_t>& count,
                     const std::function<void(const vlen&)>& body,
                     size_t unroll = 1);

    // Same loop, for a trip count known while the kernel is being built.
    // The full iterations are then emitted straight-line and the remainder
    // once, so nothing pays for a counter, a compare, a branch or a
    // per-iteration active length. LLVM reaches the same shape when SCEV
    // gives the vectorizer a constant trip count: the vector loop is fully
    // unrolled and a single epilogue handles what is left.
    //
    // Falls back to the rolled loop above `peel_limit()` full iterations,
    // since the code grows by one body per iteration.
    template <size_t N>
    void foreach_vec(size_t count, const std::function<void(const vlen&)>& body);

    // Maximum number of full iterations `foreach_vec` will emit
    // straight-line. LLVM's -force-vector-interleave / unroll thresholds;
    // 0 forces the rolled loop, which is how the two shapes are A/B'd in
    // one process. Defaults from OV_JIT_IR_PEEL.
    void set_peel_limit(size_t limit) { _peel_limit = limit; }
    [[nodiscard]] size_t peel_limit() const { return _peel_limit; }

    // Single-body loop with automatic epilogue. The body builder runs twice
    // — once for the main loop with vlen::all(), once for the tail with
    // vlen::count(width % N) — and receives its active length as an
    // argument, which it passes on to loads and stores.
    template <size_t N>
    void foreach_with_epilogue(const variable<size_t>& width,
                               const std::function<void(const vlen&)>& body);

    // ir_load from ir_ptr: the stride scales the active length, so a
    // subsampled plane (stride N/2) loads half as many elements as the
    // main plane for the same iteration. `extra_byte_offset` is added on
    // top of the pointer's accumulated displacement, for kernels that
    // address a second region off the same base (RoPE's two halves).
    //
    // Prefer these over `ir_load(p.ptr, off, vl)`: going through the raw
    // `variable<T*>` bypasses the displacement, so a peeled iteration
    // would read the first iteration's data.
    template <size_t N, typename T>
    variable<float[N]> ir_load(const ir_ptr<T>& src, size_t extra_byte_offset,
                               const vlen& vl = vlen::all());

    template <size_t N, typename T>
    variable<float[N]> ir_load(const ir_ptr<T>& src, const vlen& vl = vlen::all()) {
        return ir_load<N>(src, size_t{0}, vl);
    }

    template <size_t N, typename T, typename ElemT>
    void ir_store(const ir_ptr<T>& dst, size_t extra_byte_offset,
                  const variable<ElemT[N]>& val, const vlen& vl = vlen::all());

    template <typename T, size_t N>
    void store_interleaved3(const ir_ptr<T>& dst,
                            const variable<float[N]>& a,
                            const variable<float[N]>& b,
                            const variable<float[N]>& c,
                            const vlen& vl = vlen::all());

    // Pointer advance by one iteration's worth of elements. A no-op when
    // `vl` is a tail (the tail runs once, nothing follows it).
    //
    // In straight-line code this folds into the pointer's displacement
    // instead of emitting anything, where the target says a displacement
    // that large is free (`is_legal_access_offset`). Inside a loop body it
    // always emits the increment: the body is recorded once and runs many
    // times, so its induction update cannot be a constant. Non-const
    // because the fold mutates `ptr.disp`.
    template <typename T>
    void ir_advance(ir_ptr<T>& ptr, const vlen& vl = vlen::all());

    // Raw pointer advance with an explicit byte step.
    template <typename PtrT>
    void ir_advance(const variable<PtrT>& ptr, size_t bytes,
                    const vlen& vl = vlen::all());

    // ── IR-managed stack and memory ops (LLVM-style) ──────────

    // Allocate `size` bytes on the stack, aligned to `alignment`.
    // Returns a GPR value_id holding the stack address at lowering time.
    // All allocas are coalesced into one sub/add rsp pair in end_ir().
    // Mirrors LLVM's alloca instruction.
    jit_kernel_ir::value_id ir_alloca(size_t size, size_t alignment = 1);

    // Scalar copy: copy `count` elements of `elem_size` bytes from src to dst.
    // All three operands are GPR value_ids (IR-managed). The loop index and
    // temp register are also IR-managed — no reserve<Reg64>() inside closures.
    // Mirrors LLVM's llvm.memcpy intrinsic.
    template <typename T>
    void ir_memcpy(jit_kernel_ir::value_id dst,
                   jit_kernel_ir::value_id src,
                   jit_kernel_ir::value_id count);
};

template <>
const Xbyak::Reg64& jit_kernel::reserve<Xbyak::Reg64>();

// Raw xbyak scalar copy loop — no DSL, no IR, no foreach.
// Used by store_interleaved3 tail and runtime-length load/store.
template <typename T>
void jit_kernel::copy(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    using namespace Xbyak;
    const auto& addr_frame = address_frame(sizeof(T));
    auto p = reserve<typename reg_traits_by_size<sizeof(T)>::type>();
    auto idx = reserve<Reg64>();
    Label loop, exit;
    xor_(idx, idx);
    L(loop);
    cmp(idx, size);
    jge(exit, CodeGenerator::T_NEAR);
    mov(p, addr_frame[src + idx * sizeof(T)]);
    mov(addr_frame[dst + idx * sizeof(T)], p);
    inc(idx);
    jmp(loop, CodeGenerator::T_NEAR);
    L(exit);
    free(idx);
    free(p);
}

template <typename T>
void jit_kernel::copy(const Xbyak::Address& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    using namespace Xbyak;
    const auto& addr_frame = address_frame(sizeof(T));
    auto p = reserve<typename reg_traits_by_size<sizeof(T)>::type>();
    auto d = reserve<Reg64>();
    auto idx = reserve<Reg64>();
    lea(d, dst);
    Label loop, exit;
    xor_(idx, idx);
    L(loop);
    cmp(idx, size);
    jge(exit, CodeGenerator::T_NEAR);
    mov(p, addr_frame[src + idx * sizeof(T)]);
    mov(addr_frame[d + idx * sizeof(T)], p);
    inc(idx);
    jmp(loop, CodeGenerator::T_NEAR);
    L(exit);
    free(idx);
    free(d);
    free(p);
}

template <typename DstT, size_t N, typename SrcT>
void jit_kernel::load(const variable<DstT[N]>& dst, const variable<SrcT>& src, size_t length) {
    static_assert(std::is_same_v<typename variable<SrcT>::reg_type, const Xbyak::Reg64>,
                  "Source register must be Reg64");

    using src_type = std::remove_cv_t<std::remove_pointer_t<SrcT>>;
    using dst_type = std::remove_cv_t<std::remove_pointer_t<DstT>>;

    const std::vector<size_t> pool_vec_idxs(_free_rmmregs.begin(), _free_rmmregs.end());
    const std::vector<size_t> pool_gpr_idxs(_free_x64regs.begin(), _free_x64regs.end());

    const auto src_prc = internal::type2precision<src_type>();
    const auto dst_prc = internal::type2precision<dst_type>();

    const auto key = load_emitter_params(src_prc, dst_prc, length).hash();
    if (!_emitters[key]) {
        _emitters[key] =
            std::make_unique<jit_load_emitter>(this, internal::get_current_isa(), src_prc, dst_prc, length);
    }
    _emitters[key]->emit_code({static_cast<size_t>(static_cast<const Xbyak::Operand&>(src).getIdx())},
                              {static_cast<size_t>(static_cast<const Xbyak::Operand&>(dst).getIdx())},
                              pool_vec_idxs,
                              pool_gpr_idxs);
}

template <typename DstT, size_t N, typename SrcT>
void jit_kernel::load(const variable<DstT[N]>& dst, const variable<SrcT>& src, const variable<size_t>& length) {
    using src_type = std::remove_cv_t<std::remove_pointer_t<SrcT>>;

    auto s = stack(N * sizeof(src_type));
    s.clear();

    auto tmp = var<SrcT>();
    tmp = s.pointer();

    copy<src_type>(tmp, src, length);

    load(dst, tmp);
}

template <typename DstT, typename SrcT, size_t N>
void jit_kernel::store(const variable<DstT>& dst, const variable<SrcT[N]>& src, size_t length) {
    static_assert(std::is_same_v<typename variable<DstT>::reg_type, const Xbyak::Reg64>,
                  "Destination register must be Reg64");
    OPENVINO_ASSERT(length == N, "store: partial stores not supported, use ir_store with masking");
    ir_store(dst, src);
}

template <typename DstT, typename SrcT, size_t N>
void jit_kernel::store(const variable<DstT>& dst, const variable<SrcT[N]>& src, const variable<size_t>& length) {
    using dst_type = std::remove_cv_t<std::remove_pointer_t<DstT>>;

    auto s = stack(N * sizeof(dst_type));

    auto tmp = var<DstT>();
    tmp = s.pointer();

    store(tmp, src);

    copy<dst_type>(dst, tmp, length);
}

template <typename B, typename E, typename S>
void jit_kernel::foreach (const B& begin,
                          const E& end,
                          const std::function<void(const variable<size_t>&)>& fn,
                          const S& step) {
    using namespace Xbyak;

    if (!_ir) {
        // Eager mode: emit the loop immediately. Kernels that still emit raw
        // xbyak (cpu_convert) and stack_frame::clear() call foreach outside
        // IR mode, so the eager form has to stay.
        auto idx = var<size_t>();
        const auto& idx_reg = idx.reg();
        if constexpr (std::is_integral_v<std::decay_t<B>>) {
            mov(idx_reg, static_cast<size_t>(begin));
        } else if constexpr (std::is_base_of_v<Xbyak::Reg, std::decay_t<B>>) {
            mov(idx_reg, begin);
        } else {
            mov(idx_reg, begin.reg());
        }

        Label loop_begin;
        Label loop_end;
        L(loop_begin);
        if constexpr (std::is_integral_v<std::decay_t<E>>) {
            cmp(idx_reg, static_cast<size_t>(end));
        } else if constexpr (std::is_base_of_v<Xbyak::Reg, std::decay_t<E>>) {
            cmp(idx_reg, end);
        } else {
            cmp(idx_reg, end.reg());
        }
        jge(loop_end, T_NEAR);

        fn(idx);

        add(idx_reg, static_cast<size_t>(step));
        jmp(loop_begin, T_NEAR);
        L(loop_end);
        return;
    }

    // IR mode: the loop counter is an IR GPR value. No .reg() at recording
    // time — physical registers are resolved at lowering time via
    // EmitContext::reads.
    jit_kernel_ir::value_id idx_vid = jit_kernel_ir::invalid_value;
    if constexpr (std::is_integral_v<std::decay_t<B>>) {
        auto begin_val = static_cast<size_t>(begin);
        idx_vid = ir_def_gpr({}, [this, begin_val](const jit_kernel_ir::EmitContext& ctx) {
            mov(Reg64(ctx.def->idx), begin_val);
        }, "loop_idx");
    } else {
        auto bvid = begin.vid();
        if (bvid != jit_kernel_ir::invalid_value) {
            idx_vid = ir_def_gpr({bvid}, gpr_copy(), "loop_idx");
        } else {
            auto bi = static_cast<std::uint32_t>(begin.reg().getIdx());
            idx_vid = ir_def_gpr({}, [this, bi](const jit_kernel_ir::EmitContext& ctx) {
                mov(Reg64(ctx.def->idx), Reg64(bi));
            }, "loop_idx");
        }
    }

    // Header reads: always idx. End added if IR-managed or captured.
    std::vector<jit_kernel_ir::value_id> header_reads = {idx_vid};

    // Build compare closure. ctx.reads[0] = idx, ctx.reads[1] = end (if IR).
    std::function<void(const jit_kernel_ir::EmitContext&)> cmp_fn;
    if constexpr (std::is_integral_v<std::decay_t<E>>) {
        cmp_fn = gpr_cmp_imm(static_cast<size_t>(end));
    } else if constexpr (std::is_base_of_v<Xbyak::Reg, std::decay_t<E>>) {
        auto end_reg_idx = end.getIdx();
        cmp_fn = [this, end_reg_idx](const jit_kernel_ir::EmitContext& ctx) {
            cmp(Reg64(ctx.reads[0].idx), Reg64(end_reg_idx));
        };
    } else {
        auto evid = end.vid();
        if (evid != jit_kernel_ir::invalid_value) {
            header_reads.push_back(evid);
            cmp_fn = gpr_cmp_reg();
        } else {
            auto end_reg_idx = static_cast<std::uint32_t>(end.reg().getIdx());
            cmp_fn = [this, end_reg_idx](const jit_kernel_ir::EmitContext& ctx) {
                cmp(Reg64(ctx.reads[0].idx), Reg64(end_reg_idx));
            };
        }
    }

    auto loop_label = make_label();
    auto exit_label = make_label();
    auto step_val = static_cast<size_t>(step);
    auto idx_var = variable<size_t>(*this, idx_vid);

    _ir->loop(
        std::move(header_reads),
        // Header: place(loop); cmp(idx, end); branch(>=, exit)
        [mark = place_label(loop_label), cmp_fn, leave = branch(cond::greater_equal, exit_label)](
            const jit_kernel_ir::EmitContext& ctx) {
            mark(ctx);
            cmp_fn(ctx);
            leave(ctx);
        },
        // Body builder
        [&]() {
            fn(idx_var);

            // Footer: idx += step; branch(loop); place(exit)
            _ir->use({idx_vid},
                     [bump = gpr_bump(static_cast<std::int64_t>(step_val)),
                      again = branch_always(loop_label), mark = place_label(exit_label)](
                         const jit_kernel_ir::EmitContext& ctx) {
                         bump(ctx);
                         again(ctx);
                         mark(ctx);
                     },
                     "loop_footer");
        });
}

template <typename T>
jit_kernel::variable<T> jit_kernel::var() {
    using reg_type = typename reg_traits<T>::type;
    const auto& reg = reserve<reg_type>();
    return variable<T>(*this, internal::make_shared(reg, *this));
}

template <typename T>
jit_kernel::variable<T> jit_kernel::var(const T& val) {
    using reg_type = typename reg_traits<T>::type;
    const auto& reg = reserve<reg_type>();
    variable<T> res(*this, internal::make_shared(reg, *this));
    res = val;
    return res;
}

template <typename T, typename SrcT>
jit_kernel::variable<T> jit_kernel::var(const variable<SrcT>& src) {
    auto res = var<T>();
    load(res, src);
    return res;
}

template <typename T, typename SrcT>
jit_kernel::variable<T> jit_kernel::var(const variable<SrcT>& src, size_t length) {
    auto res = var<T>();
    load(res, src, length);
    return res;
}

template <typename T, typename SrcT>
jit_kernel::variable<T> jit_kernel::var(const variable<SrcT>& src, const variable<size_t>& length) {
    auto res = var<T>();
    load(res, src, length);
    return res;
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::fma(const variable<float[N]>& a,
                                                const variable<float[N]>& b,
                                                const variable<float[N]>& c) {
    // fma(a, b, c) = a * b + c.
    // c is the tied operand (seed for destructive FMA). The emit closure
    // handles the constraint: vmovups(def, c_reg) before the FMA. If the
    // allocator assigns def == c's register, it's a self-move (zero cost).
    return vec_op(Insn3::fmadd231ps, c, a, b);
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::fmsub(const variable<float[N]>& a,
                                                  const variable<float[N]>& b,
                                                  const variable<float[N]>& c) {
    // fmsub(a, b, c) = a * b - c.
    return vec_op(Insn3::fmsub231ps, c, a, b);
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::fnma(const variable<float[N]>& a,
                                                 const variable<float[N]>& b,
                                                 const variable<float[N]>& c) {
    // fnma(a, b, c) = c - a * b.
    return vec_op(Insn3::fnmadd231ps, c, a, b);
}

// ── IR mode: instructions as data ──────────────────────────────────────

template <typename Reg>
void jit_kernel::lower(Insn2 insn, const Reg& d, const Reg& s1, const Reg& s2) {
    switch (insn) {
    case Insn2::vaddps: uni_vaddps(d, s1, s2); break;
    case Insn2::vsubps: uni_vsubps(d, s1, s2); break;
    case Insn2::vmulps: uni_vmulps(d, s1, s2); break;
    case Insn2::vmaxps: uni_vmaxps(d, s1, s2); break;
    case Insn2::vminps: uni_vminps(d, s1, s2); break;
    default: OPENVINO_THROW("jit_kernel::lower: unknown Insn2 value ", static_cast<int>(insn));
    }
}

template <size_t N>
void jit_kernel::ir_accumulate(const variable<float[N]>& acc, Insn2 insn,
                               const variable<float[N]>& x) {
    using reg_type = typename reg_traits<float[N]>::type;
    OPENVINO_ASSERT(acc.vid() != jit_kernel_ir::invalid_value,
                    "ir_accumulate: the accumulator is not an IR value");

    _ir->def_into(acc.vid(), {acc.vid(), x.vid()},
        [this, insn](const jit_kernel_ir::EmitContext& ctx) {
            lower(insn,
                  reg_type(ctx.def->idx),
                  reg_type(ctx.reads[0].idx),
                  reg_type(ctx.reads[1].idx));
        },
        "accumulate");

    // Only the addend may come from memory. The accumulator is the
    // destination, and folding it would mean reading the running total
    // from memory and writing it to a register — a different program.
    auto& op = _ir->last();
    op.foldable_reads = 0b10;
    op.fold_emit = [this, insn](const jit_kernel_ir::EmitContext& ctx) {
        lower(insn,
              reg_type(ctx.def->idx),
              reg_type(ctx.reads[0].idx),
              address_frame(sizeof(reg_type))[Xbyak::Reg64(ctx.folded->base.idx) +
                                              ctx.folded->offset]);
    };
}

template <size_t N>
void jit_kernel::ir_accumulate(const variable<float[N]>& acc, Insn3 insn,
                               const variable<float[N]>& a,
                               const variable<float[N]>& b) {
    using reg_type = typename reg_traits<float[N]>::type;
    OPENVINO_ASSERT(acc.vid() != jit_kernel_ir::invalid_value,
                    "ir_accumulate: the accumulator is not an IR value");

    // No tie and no seed copy: the FMA is destructive in its accumulator
    // and that is exactly what is wanted here, so the post-two-address
    // form is recorded directly.
    _ir->def_into(acc.vid(), {acc.vid(), a.vid(), b.vid()},
        [this, insn](const jit_kernel_ir::EmitContext& ctx) {
            lower(insn,
                  reg_type(ctx.def->idx),
                  reg_type(ctx.reads[1].idx),
                  reg_type(ctx.reads[2].idx));
        },
        "accumulate_fma");

    // Either multiplicand may come from memory; the accumulator may not.
    auto& op = _ir->last();
    op.foldable_reads = 0b110;
    op.fold_emit = [this, insn](const jit_kernel_ir::EmitContext& ctx) {
        const auto kept = (ctx.folded->read == 1) ? 2U : 1U;
        lower(insn,
              reg_type(ctx.def->idx),
              reg_type(ctx.reads[kept].idx),
              address_frame(sizeof(reg_type))[Xbyak::Reg64(ctx.folded->base.idx) +
                                              ctx.folded->offset]);
    };
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::vec_op(Insn2 insn,
                                                   const variable<float[N]>& a,
                                                   const variable<float[N]>& b) {
    using reg_type = typename reg_traits<float[N]>::type;
    auto vid = _ir->def({a.vid(), b.vid()},
        [this, insn](const jit_kernel_ir::EmitContext& ctx) {
            lower(insn,
                  reg_type(ctx.def->idx),
                  reg_type(ctx.reads[0].idx),
                  reg_type(ctx.reads[1].idx));
        },
        "vec_op");

    // Declare which operands may come from memory, and supply the memory
    // form of the instruction — the DSL's equivalent of a target
    // description listing both rr and rm.
    //
    // The memory operand must be last on x86, so folding operand 0 means
    // swapping the sources: only sound for a commutative operation.
    // vsubps is not commutative; vmaxps/vminps are not either in the
    // strict sense, because they return the second source when an operand
    // is NaN.
    const bool commutative = (insn == Insn2::vaddps || insn == Insn2::vmulps);
    auto& op = _ir->last();
    op.foldable_reads = commutative ? 0b11 : 0b10;
    op.fold_emit = [this, insn](const jit_kernel_ir::EmitContext& ctx) {
        const auto kept = (ctx.folded->read == 0) ? 1U : 0U;
        lower(insn,
              reg_type(ctx.def->idx),
              reg_type(ctx.reads[kept].idx),
              address_frame(sizeof(reg_type))[Xbyak::Reg64(ctx.folded->base.idx) +
                                              ctx.folded->offset]);
    };
    return variable<float[N]>(*this, vid);
}

template <typename Reg>
void jit_kernel::lower(Insn2 insn, const Reg& d, const Reg& s1, const Xbyak::Address& s2) {
    switch (insn) {
    case Insn2::vaddps: uni_vaddps(d, s1, s2); break;
    case Insn2::vsubps: uni_vsubps(d, s1, s2); break;
    case Insn2::vmulps: uni_vmulps(d, s1, s2); break;
    case Insn2::vmaxps: uni_vmaxps(d, s1, s2); break;
    case Insn2::vminps: uni_vminps(d, s1, s2); break;
    default: OPENVINO_THROW("jit_kernel::lower: unknown Insn2 value ", static_cast<int>(insn));
    }
}

template <typename Reg>
void jit_kernel::lower(Insn3 insn, const Reg& d, const Reg& s1, const Reg& s2) {
    // Destructive FMA: d = d + s1 * s2 (fmadd231) or d = d - s1 * s2 (fnmadd231).
    // The seed copy (vmovups into d) is handled by vec_copy() — the allocator
    // may coalesce it, eliminating the copy entirely.
    switch (insn) {
    case Insn3::fmadd231ps:  uni_vfmadd231ps(d, s1, s2);  break;
    case Insn3::fnmadd231ps: uni_vfnmadd231ps(d, s1, s2); break;
    case Insn3::fmsub231ps:  Xbyak::CodeGenerator::vfmsub231ps(d, s1, s2); break;
    default: OPENVINO_THROW("jit_kernel::lower: unknown Insn3 value ", static_cast<int>(insn));
    }
}

template <typename Reg>
void jit_kernel::lower(Insn3 insn, const Reg& d, const Reg& s1, const Xbyak::Address& s2) {
    switch (insn) {
    case Insn3::fmadd231ps:  uni_vfmadd231ps(d, s1, s2);  break;
    case Insn3::fnmadd231ps: uni_vfnmadd231ps(d, s1, s2); break;
    case Insn3::fmsub231ps:  Xbyak::CodeGenerator::vfmsub231ps(d, s1, s2); break;
    default: OPENVINO_THROW("jit_kernel::lower: unknown Insn3 value ", static_cast<int>(insn));
    }
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::vec_op(Insn3 insn,
                                                   const variable<float[N]>& seed,
                                                   const variable<float[N]>& a,
                                                   const variable<float[N]>& b) {
    using reg_type = typename reg_traits<float[N]>::type;

    // Destructive FMA: tied_to=0 marks reads[0] as the seed.
    // TwoAddressPass inserts a COPY before this op and removes the tie.
    // After allocation, def is guaranteed to hold the seed value
    // (either coalesced or the COPY placed it there).
    auto vid = _ir->def_tied({seed.vid(), a.vid(), b.vid()}, /*tied_to=*/0,
        [this, insn](const jit_kernel_ir::EmitContext& ctx) {
            lower(insn,
                  reg_type(ctx.def->idx),
                  reg_type(ctx.reads[1].idx),
                  reg_type(ctx.reads[2].idx));
        },
        "fma");

    // The seed is the destination and cannot be memory; either
    // multiplicand can, since the product commutes.
    auto& op = _ir->last();
    op.foldable_reads = 0b110;
    op.fold_emit = [this, insn](const jit_kernel_ir::EmitContext& ctx) {
        const auto kept = (ctx.folded->read == 1) ? 2U : 1U;
        lower(insn,
              reg_type(ctx.def->idx),
              reg_type(ctx.reads[kept].idx),
              address_frame(sizeof(reg_type))[Xbyak::Reg64(ctx.folded->base.idx) +
                                              ctx.folded->offset]);
    };
    return variable<float[N]>(*this, vid);
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::vec_copy(const variable<float[N]>& src) {
    using reg_type = typename reg_traits<float[N]>::type;
    auto vid = _ir->copy(src.vid(),
        [this](const jit_kernel_ir::EmitContext& ctx) {
            uni_vmovups(reg_type(ctx.def->idx), reg_type(ctx.reads[0].idx));
        },
        "copy");
    return variable<float[N]>(*this, vid);
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::vec_permute(const variable<float[N]>& src,
                                                        const uint8_t* order) {
    using reg_type = typename reg_traits<float[N]>::type;

    if constexpr (N <= 4) {
        // SSE: shufps with imm8 — no extra register needed
        uint8_t imm8 = 0;
        for (std::size_t i = 0; i < 4; ++i)
            imm8 |= order[i] << (i * 2);
        return ir_def<N>({src.vid()},
            [this, imm8](const jit_kernel_ir::EmitContext& ctx) {
                Xbyak::Xmm def(ctx.def->idx), s(ctx.reads[0].idx);
                if (def.getIdx() != s.getIdx())
                    movdqu(def, s);
                shufps(def, s, imm8);
            },
            "shufps");
    } else {
        int data[N];
        for (std::size_t i = 0; i < N; ++i)
            data[i] = order[i];
        const int* cref = constant(data, N);
        auto addr = reinterpret_cast<std::uintptr_t>(cref);

        // The table address needs a GPR to load through. Ask the allocator
        // for one (an IR value) instead of borrowing a register and
        // push/popping around it: emit closures run after allocation, so
        // "borrowing" means clobbering whatever the allocator put there.
        auto addr_vid = _ir->def({},
            [this, addr](const jit_kernel_ir::EmitContext& ctx) {
                mov(Xbyak::Reg64(ctx.def->idx), addr);
            },
            "perm_table_addr", jit_kernel_ir::RegisterClass::GPR);
        auto table_vid = _ir->def({addr_vid},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                reg_type def(ctx.def->idx);
                uni_vmovdqu(def, address_frame(sizeof(reg_type))[Xbyak::Reg64(ctx.reads[0].idx)]);
            },
            "perm_table");
        return ir_def<N>({table_vid, src.vid()},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                reg_type def(ctx.def->idx);
                reg_type table(ctx.reads[0].idx);
                reg_type s(ctx.reads[1].idx);
                vpermps(def, table, s);
            },
            "vpermps");
    }
}

template <size_t N, typename PtrT>
jit_kernel::variable<float[N]> jit_kernel::ir_load(const variable<PtrT>& src_ptr,
                                                   size_t byte_offset,
                                                   const vlen& vl) {
    // A count with no predicate: if this target can predicate loads of
    // this element type, materialize the mask and use it; otherwise fall
    // back to scalarized access (LLVM's ScalarizeMaskedMemIntrin, done
    // here rather than as a pass because there is only one memory op form).
    if (vl.has_count() && !vl.has_mask()) {
        if (target().supports_masked_access(sizeof(elem_of<PtrT>))) {
            auto mask = ir_active_lane_mask<N>(variable<size_t>(*this, vl.count()));
            return ir_load<N>(src_ptr, byte_offset, vlen::predicated(mask, vl.count()));
        }
        return ir_load_partial<N>(src_ptr, variable<size_t>(*this, vl.count()), byte_offset);
    }

    using reg_type = typename reg_traits<float[N]>::type;
    using elem_type = std::remove_cv_t<std::remove_pointer_t<PtrT>>;

    // The pointer must be an IR value: call arg() inside IR mode. Capturing
    // a physical register index at recording time is not valid — the
    // allocator assigns registers afterwards, so the captured index refers
    // to whatever ends up there.
    auto ptr_vid = src_ptr.vid();
    OPENVINO_ASSERT(ptr_vid != jit_kernel_ir::invalid_value,
                    "ir_load: pointer is not an IR value (call arg() after begin_ir())");

    // The predicate is an operand, not ambient state: this op reads it, so
    // the allocator keeps it live and is free to place it anywhere in the
    // mask file.
    const bool masked = vl.has_mask();
    std::vector<jit_kernel_ir::value_id> reads{ptr_vid};
    if (masked) {
        reads.push_back(vl.mask());
    }

    auto emit = [this, byte_offset, masked]
                (const jit_kernel_ir::EmitContext& ctx) {
        auto ptr_reg = Xbyak::Reg64(ctx.reads[0].idx);
        auto dst = reg_type(ctx.def->idx);
        const auto mask_idx = masked ? ctx.reads[1].idx : 0U;

        if constexpr (std::is_same_v<elem_type, uint8_t>) {
            auto addr = address_frame(N)[ptr_reg + byte_offset];
            if (masked) {
                vpmovzxbd(dst | Xbyak::Opmask(mask_idx) | T_z, addr);
            } else {
                uni_vpmovzxbd(dst, addr);
            }
            uni_vcvtdq2ps(dst, dst);
        } else if constexpr (std::is_same_v<elem_type, ov::float16>) {
            auto addr = address_frame(N * sizeof(ov::float16))[ptr_reg + byte_offset];
            if (masked) {
                vcvtph2ps(dst | Xbyak::Opmask(mask_idx) | T_z, addr);
            } else {
                vcvtph2ps(dst, addr);
            }
        } else if constexpr (std::is_same_v<elem_type, ov::bfloat16>) {
            auto addr = address_frame(N * sizeof(ov::bfloat16))[ptr_reg + byte_offset];
            if (masked) {
                vpmovzxwd(dst | Xbyak::Opmask(mask_idx) | T_z, addr);
            } else {
                vpmovzxwd(dst, addr);
            }
            vpslld(dst, dst, 16);
        } else {
            auto addr = address_frame(sizeof(reg_type))[ptr_reg + byte_offset];
            if (masked) {
                vmovups(dst | Xbyak::Opmask(mask_idx) | T_z, addr);
            } else {
                uni_vmovups(dst, addr);
            }
        }
    };

    const char* name = masked ? "load_masked" : "load";
    auto vid = _ir->def(std::move(reads), std::move(emit), name);

    // Memory effects, so no pass reorders this against a store. A plain
    // f32 load is also a fold candidate: its consumer can take the address
    // directly. Converting loads are not — the conversion has no memory
    // form — and neither are masked ones.
    auto& op = _ir->last();
    op.may_load = true;
    op.mem_ptr_read = 0;
    op.mem_offset = static_cast<std::uint32_t>(byte_offset);
    if (masked || !std::is_same_v<elem_type, float>) {
        op.mem_ptr_read = -1;  // still may_load, but not foldable
        op.mem_offset = 0;
    }
    return variable<float[N]>(*this, vid);
}

template <typename PtrT>
jit_kernel_ir::value_id jit_kernel::ir_offset_ptr(const variable<PtrT>& base,
                                                  size_t byte_offset) {
    auto pvid = base.vid();
    OPENVINO_ASSERT(pvid != jit_kernel_ir::invalid_value,
                    "ir_offset_ptr: pointer is not an IR value");
    if (byte_offset == 0) {
        return pvid;
    }
    return ir_def_gpr({pvid}, gpr_offset(byte_offset), "offset_ptr");
}

template <size_t N, typename PtrT>
jit_kernel::variable<float[N]> jit_kernel::ir_load_partial(const variable<PtrT>& src_ptr,
                                                            const variable<size_t>& count,
                                                            size_t byte_offset) {
    using elem_type = std::remove_cv_t<std::remove_pointer_t<PtrT>>;

    // Allocate a zeroed stack slot, memcpy count elements into it, then
    // full-width ir_load from the slot. All IR ops — no raw xbyak.
    // Slot must be at least as wide as a full vector store (N * sizeof(float)):
    // the zero loop uses uni_vmovups which writes N floats (the hardware
    // register width), not sizeof(elem_type) * N.
    constexpr size_t elem_slot = N * sizeof(elem_type);
    constexpr size_t vec_width = N * sizeof(float);  // hardware register width in bytes
    constexpr size_t slot_bytes = elem_slot > vec_width ? elem_slot : vec_width;
    auto stack = ir_alloca(slot_bytes);
    auto stack_ptr = variable<PtrT>(*this, stack);

    // Zero the stack slot (ensures elements beyond count are zero).
    // Use an IR vec value for the zero to avoid clobbering live registers.
    auto zero = ir_def<N>({}, [this](const jit_kernel_ir::EmitContext& ctx) {
        using reg_type = typename reg_traits<float[N]>::type;
        uni_vxorps(reg_type(ctx.def->idx), reg_type(ctx.def->idx), reg_type(ctx.def->idx));
    }, "zero");

    ir_use({stack, zero.vid()}, [this, slot_bytes](const jit_kernel_ir::EmitContext& ctx) {
        using reg_type = typename reg_traits<float[N]>::type;
        auto ptr_reg = Xbyak::Reg64(ctx.reads[0].idx);
        auto zero_reg = reg_type(ctx.reads[1].idx);
        for (size_t off = 0; off < slot_bytes; off += sizeof(reg_type)) {
            uni_vmovups(ptr[ptr_reg + off], zero_reg);
        }
    }, "zero_slot");

    _ir->last().may_store = true;

    // Scalar copy of `count` elements from src (+offset) into the slot.
    ir_memcpy<elem_type>(stack, ir_offset_ptr(src_ptr, byte_offset), count.vid());

    // Full-width type-converting load from the stack — always safe, the
    // slot is N-wide and zero-filled beyond `count`.
    return ir_load<N>(stack_ptr, size_t{0}, vlen::all());
}

template <typename T>
void jit_kernel::ir_memcpy(jit_kernel_ir::value_id dst_vid,
                            jit_kernel_ir::value_id src_vid,
                            jit_kernel_ir::value_id count_vid) {
    // Two IR-managed GPR temporaries for the copy loop.
    // No reserve<Reg64>() — everything through the allocator.
    auto idx_vid = _ir->def({}, [this](const jit_kernel_ir::EmitContext& ctx) {
        xor_(Xbyak::Reg64(ctx.def->idx), Xbyak::Reg64(ctx.def->idx));
    }, "memcpy_idx", jit_kernel_ir::RegisterClass::GPR);

    auto tmp_vid = _ir->def({}, [](const jit_kernel_ir::EmitContext&) {},
        "memcpy_tmp", jit_kernel_ir::RegisterClass::GPR);

    _ir->use({dst_vid, src_vid, count_vid, idx_vid, tmp_vid},
        [this](const jit_kernel_ir::EmitContext& ctx) {
            using namespace Xbyak;
            auto dst_reg = Reg64(ctx.reads[0].idx);
            auto src_reg = Reg64(ctx.reads[1].idx);
            auto cnt_reg = Reg64(ctx.reads[2].idx);
            auto idx_reg = Reg64(ctx.reads[3].idx);
            // Use the sub-register width matching T
            constexpr size_t elem_sz = sizeof(T);

            Label loop, exit;
            L(loop);
            cmp(idx_reg, cnt_reg);
            jge(exit, CodeGenerator::T_NEAR);
            if constexpr (elem_sz == 1) {
                // Force ext8bit=true for indices 4-7 to encode spl/bpl/sil/dil
                // instead of legacy ah/ch/dh/bh. Without REX, Reg8(6) = dh
                // (high byte of rdx), which clobbers the destination pointer.
                auto idx = ctx.reads[4].idx;
                auto tmp = Reg8(static_cast<int>(idx), idx >= 4 && idx <= 7);
                mov(tmp, ptr[src_reg + idx_reg]);
                mov(ptr[dst_reg + idx_reg], tmp);
            } else {
                auto tmp = Reg32(ctx.reads[4].idx);
                const auto& af = address_frame(elem_sz);
                mov(tmp, af[src_reg + idx_reg * elem_sz]);
                mov(af[dst_reg + idx_reg * elem_sz], tmp);
            }
            inc(idx_reg);
            jmp(loop, CodeGenerator::T_NEAR);
            L(exit);
        }, "memcpy_loop");

    _ir->last().may_store = true;
}

template <size_t N, typename PtrT, typename ElemT>
void jit_kernel::ir_store_partial(const variable<PtrT>& dst_ptr,
                                  const variable<ElemT[N]>& val,
                                  const variable<size_t>& count,
                                  size_t byte_offset) {
    using dst_elem = std::remove_cv_t<std::remove_pointer_t<PtrT>>;

    // Slot must hold whichever is wider: the converted elements or one
    // hardware vector (the f32 path writes the full register width).
    constexpr size_t elem_slot = N * sizeof(dst_elem);
    constexpr size_t vec_width = N * sizeof(float);
    constexpr size_t slot_bytes = elem_slot > vec_width ? elem_slot : vec_width;

    auto stack = ir_alloca(slot_bytes);
    auto stack_ptr = variable<PtrT>(*this, stack);

    // Full-width store into the slot — always safe, the slot is N-wide.
    ir_store<N>(stack_ptr, size_t{0}, val, vlen::all());

    // Copy exactly `count` elements out to the destination (+offset).
    ir_memcpy<dst_elem>(ir_offset_ptr(dst_ptr, byte_offset), stack, count.vid());
}

template <size_t N, typename PtrT, typename ElemT>
void jit_kernel::ir_store(const variable<PtrT>& dst_ptr, size_t byte_offset,
                          const variable<ElemT[N]>& val, const vlen& vl) {
    // A count with no predicate: predicate the store if the target can,
    // otherwise scalarize. A full-width store here would overrun the
    // destination on the last iteration.
    if (vl.has_count() && !vl.has_mask()) {
        if (target().supports_masked_access(sizeof(elem_of<PtrT>))) {
            auto mask = ir_active_lane_mask<N>(variable<size_t>(*this, vl.count()));
            ir_store<N>(dst_ptr, byte_offset, val, vlen::predicated(mask, vl.count()));
            return;
        }
        ir_store_partial<N>(dst_ptr, val, variable<size_t>(*this, vl.count()), byte_offset);
        return;
    }

    using reg_type = typename reg_traits<ElemT[N]>::type;
    using dst_elem = std::remove_cv_t<std::remove_pointer_t<PtrT>>;
    const bool masked = vl.has_mask();

    // Narrowing conversions are separate IR values, not in-place edits of
    // the stored value: an op that clobbers a register it only declares as
    // a read would corrupt the value for any later use. Both conversions
    // have non-destructive three-operand encodings, so the allocator is
    // free to reuse the source register when the source dies here (that
    // makes this exactly as cheap as the in-place form used to be).
    auto src_vid = val.vid();
    if constexpr (std::is_same_v<dst_elem, uint8_t>) {
        src_vid = _ir->def({src_vid}, [this](const jit_kernel_ir::EmitContext& ctx) {
            uni_vcvtps2dq(reg_type(ctx.def->idx), reg_type(ctx.reads[0].idx));
        }, "cvt_f32_i32");
    } else if constexpr (std::is_same_v<dst_elem, ov::bfloat16>) {
        // @todo claude: use vcvtneps2bf16 where available — this truncates
        // instead of rounding to nearest even.
        src_vid = _ir->def({src_vid}, [this](const jit_kernel_ir::EmitContext& ctx) {
            vpsrld(reg_type(ctx.def->idx), reg_type(ctx.reads[0].idx), 16);
        }, "cvt_f32_bf16");
    }

    // Pointer must be an IR value, same rule as ir_load.
    // reads = {ptr, val} and, when predicated, the mask.
    auto ptr_vid = dst_ptr.vid();
    OPENVINO_ASSERT(ptr_vid != jit_kernel_ir::invalid_value,
                    "ir_store: pointer is not an IR value (call arg() after begin_ir())");

    std::vector<jit_kernel_ir::value_id> reads{ptr_vid, src_vid};
    if (masked) {
        reads.push_back(vl.mask());
    }

    ir_use(std::move(reads),
        [this, byte_offset, masked]
        (const jit_kernel_ir::EmitContext& ctx) {
            auto ptr_reg = Xbyak::Reg64(ctx.reads[0].idx);
            auto src = reg_type(ctx.reads[1].idx);
            const auto mask_idx = masked ? ctx.reads[2].idx : 0U;

            if constexpr (std::is_same_v<dst_elem, uint8_t>) {
                auto addr = address_frame(N)[ptr_reg + byte_offset];
                if (masked) {
                    vpmovusdb(addr | Xbyak::Opmask(mask_idx), src);
                } else {
                    vpmovusdb(addr, src);
                }
            } else if constexpr (std::is_same_v<dst_elem, ov::float16>) {
                auto addr = address_frame(N * sizeof(ov::float16))[ptr_reg + byte_offset];
                if (masked) {
                    vcvtps2ph(addr | Xbyak::Opmask(mask_idx), src, 0x4);
                } else {
                    vcvtps2ph(addr, src, 0x4);
                }
            } else if constexpr (std::is_same_v<dst_elem, ov::bfloat16>) {
                auto addr = address_frame(N * sizeof(ov::bfloat16))[ptr_reg + byte_offset];
                if (masked) {
                    vpmovdw(addr | Xbyak::Opmask(mask_idx), src);
                } else {
                    vpmovdw(addr, src);
                }
            } else {
                auto addr = address_frame(sizeof(reg_type))[ptr_reg + byte_offset];
                if (masked) {
                    vmovups(addr | Xbyak::Opmask(mask_idx), src);
                } else {
                    uni_vmovups(addr, src);
                }
            }
        }, masked ? "store_masked" : "store");

    // Memory effects: nothing may be folded or reordered across a store.
    auto& store_op = _ir->last();
    store_op.may_store = true;
    store_op.mem_ptr_read = 0;
    store_op.mem_offset = static_cast<std::uint32_t>(byte_offset);
}

// ── foreach_vec ────────────────────────────────────────────────────────

template <size_t N>
void jit_kernel::foreach_vec(const variable<size_t>& count,
                             const std::function<void(const vlen&)>& body,
                             size_t unroll) {
    switch (target().preferred_tail_folding()) {
    case vector_target::tail_folding::mask:
        foreach_predicated<N>(count, body, unroll);
        return;
    case vector_target::tail_folding::epilogue:
        OPENVINO_ASSERT(unroll == 1, "foreach_vec: unrolling the epilogue form is not implemented");
        foreach_with_epilogue<N>(count, body);
        return;
    case vector_target::tail_folding::length:
        // RVV: the loop header sets vl = min(remaining, VLMAX) with vsetvli
        // and the body's accesses encode nothing. Needs a RISC-V code
        // generator plus vl modelled as machine state (LLVM does this with
        // implicit VL/VTYPE operands and the RISCVInsertVSETVLI pass), so
        // it is unimplemented rather than approximated.
        OPENVINO_THROW("foreach_vec: vector-length tail folding needs a target that sets vl");
    }
    OPENVINO_THROW("foreach_vec: unknown tail folding style");
}

// ── foreach_vec, constant trip count ───────────────────────────────────
//
// `count` is a C++ value here, not a register, so the induction variable
// is a compile-time constant and everything derived from it can be folded
// away: the full iterations become straight-line code and the remainder is
// handled once, with an immediate active length.
//
// The remainder still asks the target how tails are folded — a predicate
// built from an immediate on AVX-512/SVE, a counted (scalarized) step
// where there is no predication. What it never does is *derive* the
// active length at run time, which is the per-iteration cost the rolled
// loop pays.

template <size_t N>
void jit_kernel::foreach_vec(size_t count, const std::function<void(const vlen&)>& body) {
    const size_t full = count / N;
    const size_t rem = count % N;

    // One body per iteration of code. Past the limit the rolled loop is
    // the better trade, and a dynamic count has no choice anyway.
    if (full > peel_limit()) {
        foreach_vec<N>(ir_gpr_imm(count), body);
        return;
    }

    for (size_t i = 0; i < full; ++i) {
        const lane_mask_scope masks(*this);
        // Nothing reads the pointers after the last recorded step, so its
        // advances are dead. Same contract as an epilogue tail.
        const bool terminal = (i + 1 == full) && rem == 0;
        body(vlen::all(terminal));
    }

    if (rem == 0) {
        return;
    }

    const lane_mask_scope masks(*this);
    // Consumers that cannot take a predicate (interleaved stores on x86)
    // read the count instead, so both forms carry it.
    auto count_var = ir_gpr_imm(rem);

    switch (target().preferred_tail_folding()) {
    case vector_target::tail_folding::mask:
        body(vlen::predicated(ir_const_lane_mask<N>(rem), count_var.vid(), /*terminal=*/true));
        return;
    case vector_target::tail_folding::epilogue:
        body(vlen::elements(count_var.vid(), /*terminal=*/true));
        return;
    case vector_target::tail_folding::length:
        OPENVINO_THROW("foreach_vec: vector-length tail folding needs a target that sets vl");
    }
    OPENVINO_THROW("foreach_vec: unknown tail folding style");
}

// ── foreach_predicated ─────────────────────────────────────────────────
//
// One loop, no tail: each iteration computes how many elements are left
// and hands the body that active length. This is the shape SVE and RVV use
// natively and the shape AVX-512 can emulate with a k-register, so it is
// the primary strategy — foreach_vec() selects it when the target says
// predication is available.

template <size_t N>
void jit_kernel::foreach_predicated(const variable<size_t>& total_count,
                                    const std::function<void(const vlen&)>& fn,
                                    size_t unroll) {
    auto tc_vid = total_count.vid();
    OPENVINO_ASSERT(tc_vid != jit_kernel_ir::invalid_value,
                    "foreach_predicated: count is not an IR value (call arg() after begin_ir())");

    auto remaining_vid = ir_def_gpr({tc_vid}, gpr_copy(), "remaining");

    auto remaining_var = variable<size_t>(*this, remaining_vid);
    auto iter_count_var = ir_shr(ir_add(remaining_var, N * unroll - 1),
                                 static_cast<int>(std::log2(N * unroll)));

    foreach(size_t{0}, iter_count_var, [&](const variable<size_t>&) {
        for (size_t u = 0; u < unroll; ++u) {
            // Active length of this iteration: min(remaining, N). Consumers
            // that cannot take a predicate (interleaved stores on x86) need
            // the clamped count, not the remaining total.
            //
            // Early-clobber: the expansion writes its destination before
            // consuming the count, which dies here, so the allocator must
            // not hand it the count's register.
            auto active_vid = _ir->def_early_clobber({remaining_vid}, clamped_len(N),
                                                     "active_len",
                                                     jit_kernel_ir::RegisterClass::GPR);

            // The predicate is materialized once per iteration and passed
            // to every access; on SVE this is whilelt, on RVV the loop
            // would set vl instead and the accesses would encode nothing.
            const lane_mask_scope masks(*this);
            auto mask = ir_active_lane_mask<N>(variable<size_t>(*this, active_vid));
            fn(vlen::predicated(mask, active_vid));

            ir_use({remaining_vid}, gpr_bump(-static_cast<std::int64_t>(N)), "remaining_dec");
        }
    });
}

// ── foreach_with_epilogue ─────────────────────────────────────────────
// One body, recorded twice: main loop with a full active length, tail with
// the leftover element count. The active length is an argument, so the
// body's memory operations state which one they use instead of reading
// kernel state that happens to be set.

template <size_t N>
void jit_kernel::foreach_with_epilogue(const variable<size_t>& width,
                                       const std::function<void(const vlen&)>& body) {
    using namespace Xbyak;

    auto main_count = ir_shr(width, static_cast<int>(std::logb(N)));
    auto tail_count = ir_and(width, N - 1);

    foreach(size_t{0}, main_count, [&](const variable<size_t>&) {
        const lane_mask_scope masks(*this);
        body(vlen::all());
    });

    ir_cmp(tail_count, size_t{0});
    ir_if(cond::equal, [&]() {
        const lane_mask_scope masks(*this);
        body(vlen::elements(tail_count.vid(), /*terminal=*/true));
    });
}

template <typename PtrT>
void jit_kernel::ir_advance(const variable<PtrT>& ptr, size_t bytes, const vlen& vl) {
    if (vl.is_terminal()) {
        return;  // last iteration: nothing after it reads the pointer
    }
    auto pvid = ptr.vid();
    OPENVINO_ASSERT(pvid != jit_kernel_ir::invalid_value,
                    "ir_advance: pointer is not an IR value (call arg() after begin_ir())");
    ir_use({pvid}, gpr_bump(static_cast<std::int64_t>(bytes)), "ptr_advance");
}

// ── ir_ptr overloads ──────────────────────────────────────────────────

template <size_t N, typename T>
jit_kernel::variable<float[N]> jit_kernel::ir_load(const ir_ptr<T>& src, size_t extra_byte_offset,
                                                   const vlen& vl) {
    const size_t offset = src.disp + extra_byte_offset;

    // This pointer consumes a full vector per iteration, so the loop's
    // active length applies unchanged — including its predicate, which
    // must not be re-derived.
    if (src.stride >= N) {
        return ir_load<N>(src.ptr, offset, vl);
    }

    // Otherwise the active length is scaled to this pointer's stride: a
    // plane read at N/2 elements per iteration consumes half as many
    // elements as one read at N. A per-pointer count means a per-pointer
    // predicate, so any mask in `vl` does not apply here — the scaled
    // count is passed on and the memory op decides how to realize it.
    auto count_vid = [&]() -> jit_kernel_ir::value_id {
        if (!vl.has_count()) {
            // Fewer elements than lanes, but a compile-time count.
            return ir_def_gpr({}, [this, s = src.stride](const jit_kernel_ir::EmitContext& ctx) {
                mov(Xbyak::Reg64(ctx.def->idx), s);
            }, "load_count");
        }
        if (src.stride == N) {
            return vl.count();
        }
        return ir_shr(variable<size_t>(*this, vl.count()),
                      static_cast<int>(std::log2(N / src.stride))).vid();
    }();
    return ir_load<N>(src.ptr, offset, vlen::elements(count_vid));
}

template <size_t N, typename T, typename ElemT>
void jit_kernel::ir_store(const ir_ptr<T>& dst, size_t extra_byte_offset,
                          const variable<ElemT[N]>& val, const vlen& vl) {
    ir_store(dst.ptr, dst.disp + extra_byte_offset, val, vl);
}

template <typename T>
void jit_kernel::ir_advance(ir_ptr<T>& ptr, const vlen& vl) {
    if (vl.is_terminal()) {
        return;  // last iteration: nothing after it reads the pointer
    }

    const size_t step = ptr.stride * sizeof(T);
    const size_t folded = ptr.disp + step;

    // A loop body is recorded once and executed many times, so its
    // induction update has to be a real increment. Straight-line code can
    // fold it into the following accesses instead, exactly as LLVM's
    // unroller leaves constant-offset GEPs behind and instruction
    // selection folds them into the addressing mode.
    //
    // @todo claude: the legality question is asked about the iteration
    // displacement alone. An access that adds a further fixed offset on
    // top (RoPE's second half) can therefore exceed what the target can
    // encode without this noticing. Harmless on x86, where any
    // displacement we produce is encodable; a target with a bounded
    // offset field needs the check moved to the access, which in turn
    // needs somewhere to put the increment the access cannot fold.
    if (!_ir->recording_in_loop() &&
        target().is_legal_access_offset(sizeof(T), folded / (step != 0 ? step : 1), folded)) {
        ptr.disp = folded;
        return;
    }

    // Flushing: the increment has to cover the displacement accumulated so
    // far as well, since the accesses that used it read off the old base.
    ir_advance(ptr.ptr, folded, vl);
    ptr.disp = 0;
}

template <typename T, size_t N>
void jit_kernel::store_interleaved3(const ir_ptr<T>& dst,
                                    const variable<float[N]>& a,
                                    const variable<float[N]>& b,
                                    const variable<float[N]>& c,
                                    const vlen& vl) {
    // The interleave expands into three stores at fixed sub-offsets, so it
    // takes its base as a pointer rather than a base+displacement pair.
    // ir_offset_ptr folds a zero displacement away, which is every
    // interleaved store in a rolled loop; a peeled one pays one lea
    // instead of the pointer bumps it saved.
    auto base = variable<T*>(*this, ir_offset_ptr(dst.ptr, dst.disp));

    if (vl.is_all()) {
        store_interleaved3(base, a, b, c);
        return;
    }

    OPENVINO_ASSERT(vl.has_count(),
                    "store_interleaved3: a short iteration needs an element count, "
                    "since the three output masks are derived from it");
    auto count = variable<size_t>(*this, vl.count());

    // Predicate the three stores where the target can, and only otherwise
    // fall back to building the interleave in a stack slot and copying
    // count*3 elements out — which costs ~20 GPR values and a copy loop.
    // On SVE this same call is ST3 under a governing predicate, and on RVV
    // a segment store honouring vl; on AVX-512 the interleave is built by
    // hand and the stores that write it out carry the masks.
    if constexpr (3 * N <= 64) {
        if (target().supports_masked_interleaved_access()) {
            store_interleaved3_predicated(base, a, b, c, count);
            return;
        }
    }
    store_interleaved3(base, a, b, c, count);
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::ir_def(std::vector<jit_kernel_ir::value_id> reads,
                                                   jit_kernel_ir::EmitFn emit,
                                                   const char* name) {
    auto vid = _ir->def(std::move(reads), std::move(emit), name);
    return variable<float[N]>(*this, vid);
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::ir_broadcast(const Xbyak::Address& addr) {
    using reg_type = typename reg_traits<float[N]>::type;
    return ir_def<N>({}, [this, addr](const jit_kernel_ir::EmitContext& ctx) {
        uni_vbroadcastss(reg_type(ctx.def->idx), addr);
    }, "broadcast");
}

template <size_t N, typename PtrT>
jit_kernel::variable<float[N]> jit_kernel::ir_broadcast(const variable<PtrT>& ptr,
                                                        size_t byte_offset) {
    using reg_type = typename reg_traits<float[N]>::type;
    using elem_type = std::remove_cv_t<std::remove_pointer_t<PtrT>>;
    static_assert(std::is_same_v<elem_type, float>, "ir_broadcast: float pointers only");

    auto pvid = ptr.vid();
    OPENVINO_ASSERT(pvid != jit_kernel_ir::invalid_value,
                    "ir_broadcast: pointer is not an IR value (call arg() after begin_ir())");

    return ir_def<N>({pvid}, [this, byte_offset](const jit_kernel_ir::EmitContext& ctx) {
        uni_vbroadcastss(reg_type(ctx.def->idx),
                         address_frame(sizeof(float))[Xbyak::Reg64(ctx.reads[0].idx) + byte_offset]);
    }, "broadcast");
}

// Active-lane mask for `count` leading lanes of an N-lane vector.
//
// bits = count >= N ? all-ones : (1 << count) - 1, then moved to a
// predicate register. Both steps are IR values, so the scratch GPR and the
// predicate are allocated rather than borrowed.
template <size_t N>
jit_kernel_ir::value_id jit_kernel::ir_active_lane_mask(const variable<size_t>& count) {
    static_assert(N <= 64, "active lane mask supports up to 64 lanes");

    auto count_vid = count.vid();
    OPENVINO_ASSERT(count_vid != jit_kernel_ir::invalid_value,
                    "ir_active_lane_mask: count is not an IR value");

    // One predicate per active length per body instance.
    auto cached = _lane_masks.find(count_vid);
    if (cached != _lane_masks.end()) {
        return cached->second;
    }

    auto bits_vid = ir_lane_mask_bits(count_vid, N);
    auto mask_vid = ir_def_mask({bits_vid}, mask_from_bits(N), "active_lane_mask");

    _lane_masks.emplace(count_vid, mask_vid);
    return mask_vid;
}

// Same predicate, constant-folded: with the active length known at
// kernel-build time the lane bits are an immediate, so the runtime
// cmp/bts/dec sequence collapses to one mov and the def has no reads —
// hence no early-clobber constraint either.
template <size_t N>
jit_kernel_ir::value_id jit_kernel::ir_const_lane_mask(size_t active) {
    static_assert(N <= 64, "active lane mask supports up to 64 lanes");
    OPENVINO_ASSERT(active > 0, "ir_const_lane_mask: empty predicate");

    const uint64_t all_lanes = (N == 64) ? ~uint64_t{0} : ((uint64_t{1} << N) - 1);
    const uint64_t bits = (active >= N) ? all_lanes : ((uint64_t{1} << active) - 1);

    // No reads, so no early-clobber constraint: the bits are an immediate.
    auto bits_vid = _ir->def({}, gpr_set(bits), "const_lane_mask_bits",
                             jit_kernel_ir::RegisterClass::GPR);

    return ir_def_mask({bits_vid}, mask_from_bits(N), "const_lane_mask");
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::ir_zero() {
    using reg_type = typename reg_traits<float[N]>::type;
    return ir_def<N>({}, [this](const jit_kernel_ir::EmitContext& ctx) {
        uni_vxorps(reg_type(ctx.def->idx), reg_type(ctx.def->idx), reg_type(ctx.def->idx));
    }, "zero");
}

template <typename A, typename B>
void jit_kernel::ir_cmp(const A& a, const B& b) {
    static_assert(!std::is_integral_v<std::decay_t<A>>,
                  "ir_cmp: first operand must be a register, not an immediate");

    // Support both IR-managed and eagerly-allocated operands.
    // Same dual-mode pattern as ir_load/ir_store.
    auto a_vid = a.vid();
    bool a_in_ir = (a_vid != jit_kernel_ir::invalid_value);
    auto a_idx = a_in_ir ? 0U : static_cast<std::uint32_t>(a.reg().getIdx());

    std::vector<jit_kernel_ir::value_id> reads;
    if (a_in_ir) reads.push_back(a_vid);

    if constexpr (std::is_integral_v<std::decay_t<B>>) {
        auto b_val = static_cast<size_t>(b);
        _ir->use(std::move(reads), [this, a_in_ir, a_idx, b_val](const jit_kernel_ir::EmitContext& ctx) {
            auto ar = a_in_ir ? Xbyak::Reg64(ctx.reads[0].idx) : Xbyak::Reg64(a_idx);
            cmp(ar, b_val);
        }, "cmp");
    } else {
        auto b_vid = b.vid();
        bool b_in_ir = (b_vid != jit_kernel_ir::invalid_value);
        auto b_idx = b_in_ir ? 0U : static_cast<std::uint32_t>(b.reg().getIdx());
        if (b_in_ir) reads.push_back(b_vid);
        auto b_pos = a_in_ir ? 1U : 0U;

        _ir->use(std::move(reads), [this, a_in_ir, a_idx, b_in_ir, b_idx, b_pos](
                     const jit_kernel_ir::EmitContext& ctx) {
            auto ar = a_in_ir ? Xbyak::Reg64(ctx.reads[0].idx) : Xbyak::Reg64(a_idx);
            auto br = b_in_ir ? Xbyak::Reg64(ctx.reads[b_pos].idx) : Xbyak::Reg64(b_idx);
            cmp(ar, br);
        }, "cmp");
    }
}

template <typename ThenFn, typename ElseFn>
void jit_kernel::ir_if(cond skip_on, ThenFn&& then_fn, ElseFn&& else_fn) {
    auto else_label = make_label();
    auto exit_label = make_label();

    _ir->region(branch(skip_on, else_label), [&]() { then_fn(); });

    auto enter_else = [jump = branch_always(exit_label), mark = place_label(else_label)](
                          const jit_kernel_ir::EmitContext& ctx) {
        jump(ctx);
        mark(ctx);
    };
    _ir->region(std::move(enter_else), [&]() { else_fn(); });

    _ir->use({}, place_label(exit_label), "label");
}

template <typename ThenFn>
void jit_kernel::ir_if(cond skip_on, ThenFn&& then_fn) {
    auto exit_label = make_label();
    _ir->region(branch(skip_on, exit_label), [&]() { then_fn(); });
    _ir->use({}, place_label(exit_label), "label");
}

// ── 2-way deinterleave / interleave ────────────────────────────────────
//
// ISA-specific lowering for the portable deinterleave2 / interleave2 ops.
// On ARM these would be UZP1/UZP2 and ZIP1/ZIP2 respectively.

template <size_t N>
std::pair<jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>>
jit_kernel::deinterleave2(const variable<float[N]>& a,
                          const variable<float[N]>& b) {
    // Separate even-indexed and odd-indexed elements across two vectors.
    //   a = [x0 x1 x2 x3 ...], b = [xN xN+1 ...]
    //   → evens = [x0 x2 x4 ...], odds = [x1 x3 x5 ...]
    //
    // x86 lowering (both AVX2 and AVX-512):
    //   Step 1: cross-lane permute to group 128-bit blocks
    //   Step 2: in-lane vshufps to separate even/odd within blocks
    using reg_type = typename reg_traits<float[N]>::type;

    // Step 1: cross-lane
    auto emit_cross_lane = [this](uint8_t imm) {
        return [this, imm](const jit_kernel_ir::EmitContext& ctx) {
            if constexpr (N == 8) {
                vperm2i128(reg_type(ctx.def->idx),
                           reg_type(ctx.reads[0].idx),
                           reg_type(ctx.reads[1].idx), imm);
            } else {
                vshuff32x4(reg_type(ctx.def->idx),
                           reg_type(ctx.reads[0].idx),
                           reg_type(ctx.reads[1].idx), imm);
            }
        };
    };
    auto tmp0 = ir_def<N>({a.vid(), b.vid()}, emit_cross_lane(N == 8 ? 0x20 : 0x88),
                          N == 8 ? "perm2x128" : "shuff32x4");
    auto tmp1 = ir_def<N>({a.vid(), b.vid()}, emit_cross_lane(N == 8 ? 0x31 : 0xdd),
                          N == 8 ? "perm2x128" : "shuff32x4");

    // Step 2: in-lane shuffle
    auto evens = tmp0.shuffle(tmp1, 0x88);
    auto odds  = tmp0.shuffle(tmp1, 0xdd);
    return {std::move(evens), std::move(odds)};
}

template <size_t N>
std::pair<jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>>
jit_kernel::interleave2(const variable<float[N]>& evens,
                        const variable<float[N]>& odds) {
    // Merge even/odd streams back into sequential interleaved order.
    //   evens = [e0 e1 e2 ...], odds = [o0 o1 o2 ...]
    //   → lo = [e0 o0 e1 o1 ...], hi = [eN/2 oN/2 ...]
    using reg_type = typename reg_traits<float[N]>::type;

    auto emit_unpacklo = [this](const jit_kernel_ir::EmitContext& ctx) {
        vunpcklps(reg_type(ctx.def->idx),
                  reg_type(ctx.reads[0].idx),
                  reg_type(ctx.reads[1].idx));
    };
    auto emit_unpackhi = [this](const jit_kernel_ir::EmitContext& ctx) {
        vunpckhps(reg_type(ctx.def->idx),
                  reg_type(ctx.reads[0].idx),
                  reg_type(ctx.reads[1].idx));
    };

    if constexpr (N == 8) {
        // AVX2: vunpcklps + vunpckhps + vperm2i128
        auto lo_mixed = ir_def<N>({evens.vid(), odds.vid()}, emit_unpacklo, "unpacklo");
        auto hi_mixed = ir_def<N>({evens.vid(), odds.vid()}, emit_unpackhi, "unpackhi");
        auto lo = ir_def<N>({lo_mixed.vid(), hi_mixed.vid()},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                vperm2i128(reg_type(ctx.def->idx),
                           reg_type(ctx.reads[0].idx),
                           reg_type(ctx.reads[1].idx), 0x20);
            }, "perm2x128");
        auto hi = ir_def<N>({lo_mixed.vid(), hi_mixed.vid()},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                vperm2i128(reg_type(ctx.def->idx),
                           reg_type(ctx.reads[0].idx),
                           reg_type(ctx.reads[1].idx), 0x31);
            }, "perm2x128");
        return {std::move(lo), std::move(hi)};
    } else if constexpr (N == 16) {
        // AVX-512: vpermq + vunpcklps + vunpckhps
        static const uint64_t perm_idx[] = {0, 4, 1, 5, 2, 6, 3, 7};
        auto idx = ir_broadcast<N>(ptr[constant(perm_idx, sizeof(perm_idx))]);
        auto evens_perm = ir_def<N>({idx.vid(), evens.vid()},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                vpermq(reg_type(ctx.def->idx),
                       reg_type(ctx.reads[0].idx),
                       reg_type(ctx.reads[1].idx));
            }, "permq");
        auto odds_perm = ir_def<N>({idx.vid(), odds.vid()},
            [this](const jit_kernel_ir::EmitContext& ctx) {
                vpermq(reg_type(ctx.def->idx),
                       reg_type(ctx.reads[0].idx),
                       reg_type(ctx.reads[1].idx));
            }, "permq");
        auto lo = ir_def<N>({evens_perm.vid(), odds_perm.vid()}, emit_unpacklo, "unpacklo");
        auto hi = ir_def<N>({evens_perm.vid(), odds_perm.vid()}, emit_unpackhi, "unpackhi");
        return {std::move(lo), std::move(hi)};
    } else {
        OPENVINO_THROW("interleave2: unsupported vector width N=", N);
    }
}

template <size_t N>
std::tuple<jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>, jit_kernel::variable<float[N]>>
jit_kernel::interleave_regs(const variable<float[N]>& a,
                            const variable<float[N]>& b,
                            const variable<float[N]>& c) {
    // Build the inverse of the forward mapping (i*3 + offset) % N, which is
    // what vpermps wants: for each output lane k, the source lane to pick.
    auto inverse_permutation = [](size_t offset) {
        std::array<uint8_t, N> inv{};
        for (size_t i = 0; i < N; ++i) {
            inv[(i * 3 + offset) % N] = static_cast<uint8_t>(i);
        }
        return inv;
    };

    auto ap = a.permute(inverse_permutation(0));
    auto bp = b.permute(inverse_permutation(1));
    auto cp = c.permute(inverse_permutation(2));

    // 24-bit repeating "100100..." / "010010..." patterns; each output picks
    // a phase of the cycle determined by its position.
    constexpr uint32_t blend_base_1 = 0x92492492;
    constexpr uint32_t blend_base_2 = 0x24924924;

    auto build = [&](int output_index) {
        const auto shift = (output_index * N) % 3;
        const auto mask1 = static_cast<uint16_t>(blend_base_1 >> shift);
        const auto mask2 = static_cast<uint16_t>(blend_base_2 >> shift);
        return ap.blend(bp, mask1).blend(cp, mask2);
    };

    return {build(0), build(1), build(2)};
}

template <typename T, size_t N>
void jit_kernel::store_interleaved3(const variable<T*>& dst,
                                    const variable<float[N]>& a,
                                    const variable<float[N]>& b,
                                    const variable<float[N]>& c) {
    auto [o0, o1, o2] = interleave_regs(a, b, c);

    const size_t step = N * sizeof(T);
    ir_store(dst, size_t{0}, o0);
    ir_store(dst, step, o1);
    ir_store(dst, 2 * step, o2);
}

// Predicated form: the interleave is built in registers as usual, and the
// three stores that write it out are predicated.
//
// The masks come from one computation, not three. Interleaving `count`
// elements three ways writes `3*count` consecutive output elements, so
// output element j is active iff j < 3*count — and the three stores cover
// [0,N), [N,2N) and [2N,3N). Computing the low `3*count` bits once and
// shifting by N and 2N gives all three, which is cheaper and shorter-lived
// than three separately clamped subtractions.
//
// Needs 3*N bits to fit a GPR: N=16 (AVX-512) uses 48, N=8 (AVX2) 24.
// Wider vectors fall back to the counted form, checked by the caller.
template <typename T, size_t N>
void jit_kernel::store_interleaved3_predicated(const variable<T*>& dst,
                                               const variable<float[N]>& a,
                                               const variable<float[N]>& b,
                                               const variable<float[N]>& c,
                                               const variable<size_t>& count) {
    static_assert(3 * N <= 64, "store_interleaved3_predicated: 3*N lane bits must fit a GPR");

    auto [o0, o1, o2] = interleave_regs(a, b, c);

    auto total = ir_imul(count, 3);
    auto all_bits = ir_lane_mask_bits(total.vid(), 3 * N);

    // Each slice is a fresh value: ir_shr is destructive (tied), so
    // shifting in place would consume the bits the next slice needs.
    auto slice = [this, all_bits](size_t shift) {
        if (shift == 0) {
            return ir_def_mask({all_bits}, mask_from_bits(N), "interleave_mask");
        }
        auto copy = variable<size_t>(*this, ir_def_gpr({all_bits}, gpr_copy(), "mask_bits_copy"));
        auto shifted = ir_shr(copy, static_cast<int>(shift));
        return ir_def_mask({shifted.vid()}, mask_from_bits(N), "interleave_mask");
    };

    const size_t step = N * sizeof(T);
    ir_store(dst, size_t{0}, o0, vlen::predicated(slice(0)));
    ir_store(dst, step, o1, vlen::predicated(slice(N)));
    ir_store(dst, 2 * step, o2, vlen::predicated(slice(2 * N)));
}

template <typename T, size_t N>
void jit_kernel::store_interleaved3(const variable<T*>& dst,
                                    const variable<float[N]>& a,
                                    const variable<float[N]>& b,
                                    const variable<float[N]>& c,
                                    const variable<size_t>& count) {
    auto [o0, o1, o2] = interleave_regs(a, b, c);

    constexpr size_t t_step = N * sizeof(T);
    auto stack = ir_alloca(3 * t_step);
    auto stack_ptr = variable<T*>(*this, stack);

    // Full-width stores into the slot: it is 3N-wide, and the element count
    // is applied once by the memcpy below.
    ir_store(stack_ptr, size_t{0}, o0, vlen::all());
    ir_store(stack_ptr, t_step, o1, vlen::all());
    ir_store(stack_ptr, 2 * t_step, o2, vlen::all());

    auto total = ir_imul(count, 3);

    ir_memcpy<T>(dst.vid(), stack, total.vid());
}

template <typename T>
const T& jit_kernel::constant(const T& c) {
    auto res = _consts.store(&c, sizeof c);
    return *reinterpret_cast<const T*>(res);
}

template <typename T>
const T* jit_kernel::constant(const T* c, size_t size) {
    auto res = _consts.store(c, size * sizeof(T));
    return reinterpret_cast<const T*>(res);
}

template <typename T>
jit_kernel::if_expression<T> jit_kernel::_if(const boolean_expression<T>& expr) const {
    return if_expression<T>(expr);
}

namespace internal {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// shared_reg

template <typename Reg>
shared_reg<Reg> make_shared(Reg& reg, jit_kernel& kernel) {
    std::shared_ptr<Reg> ptr(&reg, [&kernel](Reg* preg) {
        kernel.free(*preg);
    });
    return ptr;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// boolean_expression

template <typename T>
boolean_expression<T>::boolean_expression(jit_kernel& kernel,
                                          type t,
                                          const shared_reg<reg_type>& lhs,
                                          const shared_reg<reg_type>& rhs)
    : _kernel(kernel),
      _type(t),
      _lhs(lhs),
      _rhs(rhs),
      _rvalue{} {}

template <typename T>
boolean_expression<T>::boolean_expression(jit_kernel& kernel, type t, const shared_reg<reg_type>& lhs, T rhs)
    : _kernel(kernel),
      _type(t),
      _lhs(lhs),
      _rvalue(rhs) {}

template <typename T>
void boolean_expression<T>::cmp(const Xbyak::Label& exit) const {
    if (_rhs) {
        _kernel.cmp(*_lhs, *_rhs);
    } else {
        _kernel.cmp(*_lhs, _rvalue);
    }

    switch (_type) {
    case type::eq: {
        _kernel.jne(exit, Xbyak::CodeGenerator::T_NEAR);
        break;
    }
    case type::neq: {
        _kernel.je(exit, Xbyak::CodeGenerator::T_NEAR);
        break;
    }
    case type::ls: {
        _kernel.jge(exit, Xbyak::CodeGenerator::T_NEAR);
        break;
    }
    case type::gt: {
        _kernel.jle(exit, Xbyak::CodeGenerator::T_NEAR);
        break;
    }
    case type::le: {
        _kernel.jg(exit, Xbyak::CodeGenerator::T_NEAR);
        break;
    }
    case type::ge: {
        _kernel.jl(exit, Xbyak::CodeGenerator::T_NEAR);
        break;
    }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// then_expression

template <typename T>
then_expression<T>::then_expression(if_expression<T>& expr) : _if_expr(expr) {}

template <typename T>
template <typename F>
void then_expression<T>::_else(F&& fn) {
    std::forward<F>(fn)();
    _if_expr._expr._kernel.L(_if_expr._exit);
    _if_expr._is_exit_valid = true;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// variable

template <typename T>
variable_base<T, register_tag>::variable_base(jit_kernel& krnl, const shared_reg<reg_type>& reg)
    : _kernel(krnl),
      _reg(reg) {}

template <typename T>
variable_base<T, register_tag>::variable_base(jit_kernel& krnl, jit_kernel_ir::value_id vid)
    : _kernel(krnl),
      _vid(vid) {}

template <typename T>
variable_base<T, register_tag>::variable_base(const variable_base& rhs) : _kernel(rhs._kernel),
                                                                          _reg(rhs._reg),
                                                                          _vid(rhs._vid) {}

template <typename T>
variable_base<T, register_tag>::variable_base(variable_base&& rhs) noexcept
    : _kernel(rhs._kernel),
      _reg(std::move(rhs._reg)),
      _vid(rhs._vid) {}

template <typename T>
variable_base<T, memory_tag>::variable_base(jit_kernel& krnl, const shared_reg<reg_type>& addr)
    : _kernel(krnl),
      _addr(addr) {}

template <typename T>
variable_base<T, memory_tag>::variable_base(const variable_base& rhs) : _kernel(rhs._kernel),
                                                                        _addr(rhs._addr) {}

template <typename T>
variable_base<T, memory_tag>::variable_base(variable_base&& rhs) noexcept
    : _kernel(rhs._kernel),
      _addr(std::move(rhs._addr)) {}

template <typename T>
variable<T, register_tag>::variable(jit_kernel& krnl)
    : base(krnl, make_shared(krnl.reserve<typename reg_traits<T>::type>(), krnl)) {}

template <typename T>
variable<T, register_tag>::variable(jit_kernel& krnl, const shared_reg<reg_type>& reg) : base(krnl, reg) {}

template <typename T>
variable<T, memory_tag>::variable(jit_kernel& krnl, const shared_reg<reg_type>& reg) : base(krnl, reg) {}

template <typename T>
const variable<T, memory_tag>& variable<T, memory_tag>::operator=(const variable<T, register_tag>& rhs) const {
    const auto& addr_frame = base::_kernel.address_frame(sizeof(T));
    base::_kernel.mov(addr_frame[base::reg()], rhs);
    return *this;
}

template <typename T, size_t N>
variable<T[N], register_tag>::variable(jit_kernel& krnl)
    : base(krnl, make_shared(krnl.reserve<typename reg_traits<T[N]>::type>(), krnl)) {}

template <typename T, size_t N>
variable<T[N], register_tag>::variable(jit_kernel& krnl, const shared_reg<reg_type>& reg) : base(krnl, reg) {}

template <typename T, size_t N>
variable<T[N], register_tag>::variable(jit_kernel& krnl, jit_kernel_ir::value_id vid) : base(krnl, vid) {}

// NOLINTEND(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)

}  // namespace internal

}  // namespace ov::intel_cpu
