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
#include "jit_kernel_ir.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

// Instruction tags for the "instructions as data" dispatch.
// Each enum value maps to a single xbyak instruction in jit_kernel::lower().
// Adding a new instruction: add enum value here + one case in lower().
enum class Insn2 : uint8_t { vaddps, vsubps, vmulps, vmaxps, vminps, COUNT };
enum class Insn3 : uint8_t { fmadd231ps, fnmadd231ps, COUNT };

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
    jit_kernel_ir::value_id _vid = jit_kernel_ir::invalid_value;
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
        if (base::_kernel.ir_mode()) {
            using reg_type = typename reg_traits<type>::type;
            return base::_kernel.template ir_def<N>(
                {base::vid(), rhs.vid()},
                [&k = base::_kernel, mask](const jit_kernel_ir::EmitContext& ctx) {
                    k.uni_vblendps(reg_type(ctx.def->idx),
                                   reg_type(ctx.reads[0].idx),
                                   reg_type(ctx.reads[1].idx), mask);
                });
        }
        variable res(base::_kernel);
        base::_kernel.uni_vblendps(res, base::reg(), rhs, mask);
        return res;
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
        if (base::_kernel.ir_mode()) {
            using reg_type = typename reg_traits<type>::type;
            // Copy order data — the pointer may not survive to lowering time.
            auto order_copy = std::make_shared<std::array<uint8_t, N>>();
            std::copy_n(order, N, order_copy->data());
            return base::_kernel.template ir_def<N>(
                {base::vid()},
                [&k = base::_kernel, order_copy](const jit_kernel_ir::EmitContext& ctx) {
                    k.uni_vpermps(reg_type(ctx.def->idx),
                                  order_copy->data(),
                                  reg_type(ctx.reads[0].idx));
                });
        }
        variable res(base::_kernel);
        base::_kernel.uni_vpermps(res, order, base::reg());
        return res;
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
        if (base::_kernel.ir_mode()) {
            using reg_type = typename reg_traits<type>::type;
            return base::_kernel.template ir_def<N>(
                {base::vid(), other.vid()},
                [&k = base::_kernel, imm](const jit_kernel_ir::EmitContext& ctx) {
                    k.uni_vshufps(reg_type(ctx.def->idx),
                                  reg_type(ctx.reads[0].idx),
                                  reg_type(ctx.reads[1].idx), imm);
                });
        }
        variable res(base::_kernel);
        base::_kernel.uni_vshufps(res, base::reg(), other, imm);
        return res;
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

struct jit_kernel : public dnnl::impl::cpu::x64::jit_generator_t {
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

    template <typename T, typename U>
    Xbyak::Address argPtr(U T::*member) const {
        auto memPtr = &(reinterpret_cast<const T*>(0)->*member);
        const size_t offs = reinterpret_cast<const char*>(memPtr) - reinterpret_cast<const char*>(0);
        return address_frame(sizeof(U))[param1 + offs];
    }

    template <typename T, typename U>
    variable<U> arg(U T::*member) {
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
    // `jcc` is the jump type for the ELSE case (e.g. jge, jne).
    template <typename ThenFn, typename ElseFn>
    void ir_if(void (Xbyak::CodeGenerator::*jcc)(const Xbyak::Label&, Xbyak::CodeGenerator::LabelType),
               ThenFn&& then_fn, ElseFn&& else_fn);

    // Overload without else branch.
    template <typename ThenFn>
    void ir_if(void (Xbyak::CodeGenerator::*jcc)(const Xbyak::Label&, Xbyak::CodeGenerator::LabelType),
               ThenFn&& then_fn);

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

    [[nodiscard]] bool ir_mode() const noexcept { return _ir_mode; }
    void begin_ir();
    void end_ir();

    // Generic dispatch: binary vector op. Handles IR/eager branching,
    // register allocation, and width dispatch in one place.
    template <size_t N>
    variable<float[N]> vec_op(Insn2 insn,
                              const variable<float[N]>& a,
                              const variable<float[N]>& b);

    // IR-mode load/store for f32 vectors. The pointer variable is an
    // eager-mode Reg64; the vector result/source is an IR value_id.
    // These emit vmovups — no type conversion. Type-converting loads
    // (u8→f32 etc.) require the barrier mechanism (not yet implemented).
    template <size_t N, typename PtrT>
    variable<float[N]> ir_load(const variable<PtrT>& ptr);

    template <size_t N, typename PtrT, typename ElemT>
    void ir_store(const variable<PtrT>& ptr, const variable<ElemT[N]>& val);

    // ir_store with byte offset from pointer base.
    template <size_t N, typename PtrT, typename ElemT>
    void ir_store(const variable<PtrT>& ptr, size_t byte_offset,
                  const variable<ElemT[N]>& val);

    // Low-level bridge helpers for IR mode: record a def/use op with a
    // custom emit closure. Used by ir_load/ir_store internally and
    // available for custom bridging when needed.
    template <size_t N>
    variable<float[N]> ir_def(std::vector<jit_kernel_ir::value_id> reads,
                              jit_kernel_ir::EmitFn emit);
    void ir_use(std::vector<jit_kernel_ir::value_id> reads,
                jit_kernel_ir::EmitFn emit);

    // Lowering: maps instruction enum to xbyak call. The only code that
    // names specific xbyak instructions. Templated on register type so
    // uni_* overloads resolve naturally from the caller's width.
    template <typename Reg>
    void lower(Insn2 insn, const Reg& d, const Reg& s1, const Reg& s2);

    // Ternary dispatch: result = f(a, b, c). FMA instructions are
    // destructive (overwrite first operand), so lower() emits a vmovups
    // seed + the FMA. The allocator's coalescing pass can eliminate the
    // seed when the source interval ends at the FMA.
    template <size_t N>
    variable<float[N]> vec_op(Insn3 insn,
                              const variable<float[N]>& a,
                              const variable<float[N]>& b,
                              const variable<float[N]>& c);

    template <typename Reg>
    void lower(Insn3 insn, const Reg& d, const Reg& s1, const Reg& s2, const Reg& s3);

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

    // Broadcast a scalar from memory into an IR vector variable.
    template <size_t N>
    variable<float[N]> ir_broadcast(const Xbyak::Address& addr);

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

    // IR mode state
    bool _ir_mode = false;
    std::unique_ptr<jit_kernel_ir::IR> _ir;
};

template <>
const Xbyak::Reg64& jit_kernel::reserve<Xbyak::Reg64>();

template <typename T>
void jit_kernel::copy(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    const auto& addr_frame = address_frame(sizeof(T));
    auto p = reserve<typename reg_traits_by_size<sizeof(T)>::type>();
    foreach (0, size, [&](const Xbyak::Reg64& idx) {
        mov(p, addr_frame[src + idx * sizeof(T)]);
        mov(addr_frame[dst + idx * sizeof(T)], p);
    })
        ;
    free(p);
}

template <typename T>
void jit_kernel::copy(const Xbyak::Address& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    const auto& addr_frame = address_frame(sizeof(T));
    auto p = reserve<typename reg_traits_by_size<sizeof(T)>::type>();
    auto d = reserve<Xbyak::Reg64>();
    lea(d, dst);
    foreach (0, size, [&](const Xbyak::Reg64& idx) {
        mov(p, addr_frame[src + idx * sizeof(T)]);
        mov(addr_frame[d + idx * sizeof(T)], p);
    })
        ;
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

    using src_type = std::remove_cv_t<std::remove_pointer_t<SrcT>>;
    using dst_type = std::remove_cv_t<std::remove_pointer_t<DstT>>;

    if (_ir_mode) {
        // @todo claude: IR mode only supports same-type full-width stores.
        // Type-converting stores (f32→u8) need the barrier mechanism.
        OPENVINO_ASSERT((std::is_same_v<src_type, dst_type>),
                        "IR mode store: type conversion not yet supported");
        OPENVINO_ASSERT(length == N, "IR mode store: partial stores not yet supported");
        ir_store(dst, src);
        return;
    }

    const std::vector<size_t> pool_vec_idxs(_free_rmmregs.begin(), _free_rmmregs.end());
    const std::vector<size_t> pool_gpr_idxs(_free_x64regs.begin(), _free_x64regs.end());

    const auto src_prc = internal::type2precision<src_type>();
    const auto dst_prc = internal::type2precision<dst_type>();

    const auto key = store_emitter_params(src_prc, dst_prc, length).hash();
    if (!_emitters[key]) {
        _emitters[key] =
            std::make_unique<jit_store_emitter>(this, internal::get_current_isa(), src_prc, dst_prc, length);
    }
    _emitters[key]->emit_code({static_cast<size_t>(static_cast<const Xbyak::Operand&>(src).getIdx())},
                              {static_cast<size_t>(static_cast<const Xbyak::Operand&>(dst).getIdx())},
                              pool_vec_idxs,
                              pool_gpr_idxs);
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

    if (!_ir_mode) {
        // Eager path: emit loop control flow immediately
        Label loop, exit;
        auto idx = var<size_t>();
        idx = begin;
        L(loop);
        cmp(idx, end);
        jge(exit, T_NEAR);
        fn(idx);
        add(idx, step);
        jmp(loop, T_NEAR);
        L(exit);
        return;
    }

    // IR path: loop counter lives in eager mode (GPR, not IR-managed).
    // Body ops record into a nested IR via _ir->loop().
    auto idx = var<size_t>();
    idx = begin;
    auto idx_reg_idx = idx.reg().getIdx();

    // Build a compare closure, handling immediate, Reg64, and variable end values.
    std::function<void()> cmp_fn;
    if constexpr (std::is_integral_v<std::decay_t<E>>) {
        auto end_val = static_cast<size_t>(end);
        cmp_fn = [this, idx_reg_idx, end_val]() {
            cmp(Xbyak::Reg64(idx_reg_idx), end_val);
        };
    } else if constexpr (std::is_base_of_v<Xbyak::Reg, std::decay_t<E>>) {
        auto end_reg_idx = end.getIdx();
        cmp_fn = [this, idx_reg_idx, end_reg_idx]() {
            cmp(Xbyak::Reg64(idx_reg_idx), Xbyak::Reg64(end_reg_idx));
        };
    } else {
        auto end_reg_idx = end.reg().getIdx();
        cmp_fn = [this, idx_reg_idx, end_reg_idx]() {
            cmp(Xbyak::Reg64(idx_reg_idx), Xbyak::Reg64(end_reg_idx));
        };
    }

    auto loop_label = std::make_shared<Xbyak::Label>();
    auto exit_label = std::make_shared<Xbyak::Label>();
    auto step_val = static_cast<size_t>(step);

    // The loop op's emit closure emits: header, then the body is lowered
    // by the lowering pass, then the footer (recorded as last op in body).
    _ir->loop(
        // Header emit: L(loop); cmp(idx, end); jge(exit)
        [this, loop_label, exit_label, cmp_fn](const jit_kernel_ir::EmitContext&) {
            L(*loop_label);
            cmp_fn();
            jge(*exit_label, Xbyak::CodeGenerator::T_NEAR);
        },
        // Body builder: records ops into nested IR
        [&]() {
            fn(idx);

            // Footer: add(idx, step); jmp(loop); L(exit)
            // Recorded as the last use() in the body so lowering emits it
            // after all body ops.
            _ir->use({}, [this, loop_label, exit_label, idx_reg_idx, step_val](
                             const jit_kernel_ir::EmitContext&) {
                add(Xbyak::Reg64(idx_reg_idx), step_val);
                jmp(*loop_label, Xbyak::CodeGenerator::T_NEAR);
                L(*exit_label);
            });
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
    // fma(a, b, c) = a * b + c. Maps to vfmadd231ps: dst = dst + s2 * s3.
    // lower() seeds dst with c (the addend), then FMAs with a, b.
    return vec_op(Insn3::fmadd231ps, c, a, b);
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::fnma(const variable<float[N]>& a,
                                                 const variable<float[N]>& b,
                                                 const variable<float[N]>& c) {
    // fnma(a, b, c) = c - a * b. Maps to vfnmadd231ps: dst = dst - s2 * s3.
    // lower() seeds dst with c (the minuend), then FNMAs with a, b.
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
jit_kernel::variable<float[N]> jit_kernel::vec_op(Insn2 insn,
                                                   const variable<float[N]>& a,
                                                   const variable<float[N]>& b) {
    using reg_type = typename reg_traits<float[N]>::type;

    if (_ir_mode) {
        auto vid = _ir->def({a.vid(), b.vid()},
            [this, insn](const jit_kernel_ir::EmitContext& ctx) {
                lower(insn,
                      reg_type(ctx.def->idx),
                      reg_type(ctx.reads[0].idx),
                      reg_type(ctx.reads[1].idx));
            });
        return variable<float[N]>(*this, vid);
    }

    // Eager path: allocate register, emit immediately
    variable<float[N]> res(*this);
    lower(insn,
          static_cast<const reg_type&>(res.reg()),
          static_cast<const reg_type&>(a.reg()),
          static_cast<const reg_type&>(b.reg()));
    return res;
}

template <typename Reg>
void jit_kernel::lower(Insn3 insn, const Reg& d, const Reg& s1, const Reg& s2, const Reg& s3) {
    // FMA semantics: result = s1 * s2 + s3 (fmadd231) or s3 - s1 * s2 (fnmadd231).
    // xbyak's vfmadd231ps overwrites the first operand: d = d + s2 * s3.
    // So we seed d with s1 (the addend), then FMA with the multiplicands.
    if (d.getIdx() != s1.getIdx()) {
        uni_vmovups(d, s1);
    }
    switch (insn) {
    case Insn3::fmadd231ps:  uni_vfmadd231ps(d, s2, s3);  break;
    case Insn3::fnmadd231ps: uni_vfnmadd231ps(d, s2, s3); break;
    default: OPENVINO_THROW("jit_kernel::lower: unknown Insn3 value ", static_cast<int>(insn));
    }
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::vec_op(Insn3 insn,
                                                   const variable<float[N]>& a,
                                                   const variable<float[N]>& b,
                                                   const variable<float[N]>& c) {
    using reg_type = typename reg_traits<float[N]>::type;

    if (_ir_mode) {
        auto vid = _ir->def({a.vid(), b.vid(), c.vid()},
            [this, insn](const jit_kernel_ir::EmitContext& ctx) {
                lower(insn,
                      reg_type(ctx.def->idx),
                      reg_type(ctx.reads[0].idx),
                      reg_type(ctx.reads[1].idx),
                      reg_type(ctx.reads[2].idx));
            });
        return variable<float[N]>(*this, vid);
    }

    // Eager path
    variable<float[N]> res(*this);
    lower(insn,
          static_cast<const reg_type&>(res.reg()),
          static_cast<const reg_type&>(a.reg()),
          static_cast<const reg_type&>(b.reg()),
          static_cast<const reg_type&>(c.reg()));
    return res;
}

template <size_t N, typename PtrT>
jit_kernel::variable<float[N]> jit_kernel::ir_load(const variable<PtrT>& src_ptr) {
    using reg_type = typename reg_traits<float[N]>::type;
    auto ptr_idx = static_cast<std::uint32_t>(src_ptr.reg().getIdx());
    return ir_def<N>({}, [this, ptr_idx](const jit_kernel_ir::EmitContext& ctx) {
        uni_vmovups(reg_type(ctx.def->idx),
                    address_frame(sizeof(reg_type))[Xbyak::Reg64(ptr_idx)]);
    });
}

template <size_t N, typename PtrT, typename ElemT>
void jit_kernel::ir_store(const variable<PtrT>& dst_ptr, const variable<ElemT[N]>& val) {
    ir_store(dst_ptr, size_t{0}, val);
}

template <size_t N, typename PtrT, typename ElemT>
void jit_kernel::ir_store(const variable<PtrT>& dst_ptr, size_t byte_offset,
                          const variable<ElemT[N]>& val) {
    using reg_type = typename reg_traits<ElemT[N]>::type;
    auto ptr_idx = static_cast<std::uint32_t>(dst_ptr.reg().getIdx());
    if (_ir_mode) {
        ir_use({val.vid()}, [this, ptr_idx, byte_offset](const jit_kernel_ir::EmitContext& ctx) {
            uni_vmovups(address_frame(sizeof(reg_type))[Xbyak::Reg64(ptr_idx) + byte_offset],
                        reg_type(ctx.reads[0].idx));
        });
    } else {
        uni_vmovups(address_frame(sizeof(reg_type))[Xbyak::Reg64(ptr_idx) + byte_offset],
                    static_cast<const reg_type&>(val.reg()));
    }
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::ir_def(std::vector<jit_kernel_ir::value_id> reads,
                                                   jit_kernel_ir::EmitFn emit) {
    if (_ir_mode) {
        auto vid = _ir->def(std::move(reads), std::move(emit));
        return variable<float[N]>(*this, vid);
    }
    // Eager: call the emit closure immediately with a fresh register.
    variable<float[N]> res(*this);
    std::vector<jit_kernel_ir::PhysReg> no_reads;
    jit_kernel_ir::EmitContext ctx{jit_kernel_ir::PhysReg{static_cast<std::uint32_t>(res.reg().getIdx())}, no_reads};
    emit(ctx);
    return res;
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::ir_broadcast(const Xbyak::Address& addr) {
    using reg_type = typename reg_traits<float[N]>::type;
    if (_ir_mode) {
        return ir_def<N>({}, [this, addr](const jit_kernel_ir::EmitContext& ctx) {
            uni_vbroadcastss(reg_type(ctx.def->idx), addr);
        });
    }
    variable<float[N]> res(*this);
    uni_vbroadcastss(res, addr);
    return res;
}

template <typename A, typename B>
void jit_kernel::ir_cmp(const A& a, const B& b) {
    // x86 cmp: first operand is always a register.
    static_assert(!std::is_integral_v<std::decay_t<A>>,
                  "ir_cmp: first operand must be a register, not an immediate");
    auto a_idx = a.reg().getIdx();
    if (_ir_mode) {
        if constexpr (std::is_integral_v<std::decay_t<B>>) {
            auto b_val = static_cast<size_t>(b);
            _ir->use({}, [this, a_idx, b_val](const jit_kernel_ir::EmitContext&) {
                cmp(Xbyak::Reg64(a_idx), b_val);
            });
        } else {
            auto b_idx = b.reg().getIdx();
            _ir->use({}, [this, a_idx, b_idx](const jit_kernel_ir::EmitContext&) {
                cmp(Xbyak::Reg64(a_idx), Xbyak::Reg64(b_idx));
            });
        }
    } else {
        cmp(a, b);
    }
}

template <typename ThenFn, typename ElseFn>
void jit_kernel::ir_if(
        void (Xbyak::CodeGenerator::*jcc)(const Xbyak::Label&, Xbyak::CodeGenerator::LabelType),
        ThenFn&& then_fn, ElseFn&& else_fn) {
    if (_ir_mode) {
        auto else_label = std::make_shared<Xbyak::Label>();
        auto exit_label = std::make_shared<Xbyak::Label>();

        _ir->region(
            [this, jcc, else_label](const jit_kernel_ir::EmitContext&) {
                (this->*jcc)(*else_label, Xbyak::CodeGenerator::T_NEAR);
            },
            [&]() { then_fn(); });

        _ir->region(
            [this, else_label, exit_label](const jit_kernel_ir::EmitContext&) {
                jmp(*exit_label, Xbyak::CodeGenerator::T_NEAR);
                L(*else_label);
            },
            [&]() { else_fn(); });

        _ir->use({}, [this, exit_label](const jit_kernel_ir::EmitContext&) {
            L(*exit_label);
        });
    } else {
        Xbyak::Label else_label, exit_label;
        (this->*jcc)(else_label, Xbyak::CodeGenerator::T_NEAR);
        then_fn();
        jmp(exit_label, Xbyak::CodeGenerator::T_NEAR);
        L(else_label);
        else_fn();
        L(exit_label);
    }
}

template <typename ThenFn>
void jit_kernel::ir_if(
        void (Xbyak::CodeGenerator::*jcc)(const Xbyak::Label&, Xbyak::CodeGenerator::LabelType),
        ThenFn&& then_fn) {
    if (_ir_mode) {
        auto exit_label = std::make_shared<Xbyak::Label>();

        _ir->region(
            [this, jcc, exit_label](const jit_kernel_ir::EmitContext&) {
                (this->*jcc)(*exit_label, Xbyak::CodeGenerator::T_NEAR);
            },
            [&]() { then_fn(); });

        _ir->use({}, [this, exit_label](const jit_kernel_ir::EmitContext&) {
            L(*exit_label);
        });
    } else {
        Xbyak::Label exit_label;
        (this->*jcc)(exit_label, Xbyak::CodeGenerator::T_NEAR);
        then_fn();
        L(exit_label);
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

template <typename T, size_t N>
void jit_kernel::store_interleaved3(const variable<T*>& dst,
                                    const variable<float[N]>& a,
                                    const variable<float[N]>& b,
                                    const variable<float[N]>& c,
                                    const variable<size_t>& count) {
    // Portable tail lowering: full-width interleave into a stack slot, then
    // byte-copy `count * 3` elements out. No ISA (except RVV via vl) has a
    // masked variable-length interleaved store, so this is the fallback every
    // non-RVV backend can reuse. A NEON port keeps the shape — just emits
    // VST3 into the slot instead of the shuffle chain. An RVV port would
    // override to `vsetvl count; vsseg3e<bits>.v v0, (dst)` and skip the slot.
    const size_t step = N * sizeof(T);
    auto slot = stack(3 * step);

    auto [o0, o1, o2] = interleave_regs(a, b, c);

    auto sp = var<T*>();
    sp = slot.pointer();
    store(sp, o0);
    sp += step;
    store(sp, o1);
    sp += step;
    store(sp, o2);

    auto bytes = count * static_cast<size_t>(3U);
    copy<T>(ptr[dst], slot.pointer(), bytes);
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
