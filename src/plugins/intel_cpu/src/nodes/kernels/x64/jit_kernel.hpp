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
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

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

    // XByak relies on implicit conversions
    operator reg_type&() const {  // NOLINT(google-explicit-constructor)
        return reg();
    }

    operator Xbyak::RegExp() const {  // NOLINT(google-explicit-constructor)
        return reg();
    }

protected:
    variable_base(jit_kernel& krnl, const shared_reg<reg_type>& reg);
    ~variable_base() = default;

    jit_kernel& _kernel;
    // mutable so const-qualified move-assignment can rebind the underlying
    // register. `variable<>` is modeled as a const handle to a mutable
    // register slot: existing `operator=(const variable&)` already writes
    // into `*_reg` through `const` methods; move-assignment extends this by
    // letting the handle point to a different slot without copying contents.
    mutable shared_reg<reg_type> _reg;
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
        variable res(base::_kernel);
        base::_kernel.uni_vblendps(res, base::reg(), rhs, mask);
        return res;
    }

    // Lane permutation (vpermps). Returns a fresh variable holding the
    // permuted contents; source is unchanged. By-value is the DSL convention
    // for all value-transforming operations — see "Value-transforming
    // operations are by value" in jit_kernel.md.
    variable permute(const std::array<uint8_t, N>& order) const {
        static_assert(std::is_same_v<T, float>, "vector permute requires float element type");
        variable res(base::_kernel);
        base::_kernel.uni_vpermps(res, order.data(), base::reg());
        return res;
    }

    variable permute(const uint8_t* order) const {
        static_assert(std::is_same_v<T, float>, "vector permute requires float element type");
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
        static_assert(std::is_same_v<T, float>, "vector shuffle requires float element type");
        variable res(base::_kernel);
        base::_kernel.uni_vshufps(res, base::reg(), base::reg(), imm);
        return res;
    }

    variable shuffle(const variable& other, uint8_t imm) const {
        static_assert(std::is_same_v<T, float>, "vector shuffle requires float element type");
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
        variable res(base::_kernel);
        base::_kernel.uni_vmaxps(res, base::reg(), lo);
        base::_kernel.uni_vminps(res, res, hi);
        return res;
    }

    // Vector arithmetic (Phase 1). Rule: one overload = one instruction.
    // Float-only; integer vector ops and divide are intentionally not provided.
    template <typename U = T>
    variable operator+(const variable& rhs) const {
        static_assert(std::is_same_v<U, float>, "vector operator+ requires float element type");
        variable res(base::_kernel);
        base::_kernel.uni_vaddps(res, base::reg(), rhs);
        return res;
    }

    template <typename U = T>
    variable operator-(const variable& rhs) const {
        static_assert(std::is_same_v<U, float>, "vector operator- requires float element type");
        variable res(base::_kernel);
        base::_kernel.uni_vsubps(res, base::reg(), rhs);
        return res;
    }

    template <typename U = T>
    variable operator*(const variable& rhs) const {
        static_assert(std::is_same_v<U, float>, "vector operator* requires float element type");
        variable res(base::_kernel);
        base::_kernel.uni_vmulps(res, base::reg(), rhs);
        return res;
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

    void uni_vpermps(const Xbyak::Xmm& x1, const uint8_t mask[4], const Xbyak::Operand& op);
    void uni_vpermps(const Xbyak::Ymm& y1, const uint8_t mask[8], const Xbyak::Operand& op);
    void uni_vpermps(const Xbyak::Zmm& z1, const uint8_t mask[16], const Xbyak::Operand& op);
    void uni_vblendps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, uint16_t mask);
    void uni_vblendps(const Xbyak::Ymm& y1, const Xbyak::Ymm& y2, uint16_t mask);
    void uni_vblendps(const Xbyak::Zmm& z1, const Xbyak::Zmm& z2, uint16_t mask);
    void uni_vblendps(const Xbyak::Xmm& dst, const Xbyak::Xmm& src1, const Xbyak::Xmm& src2, uint16_t mask);
    void uni_vblendps(const Xbyak::Ymm& dst, const Xbyak::Ymm& src1, const Xbyak::Ymm& src2, uint16_t mask);
    void uni_vblendps(const Xbyak::Zmm& dst, const Xbyak::Zmm& src1, const Xbyak::Zmm& src2, uint16_t mask);

    void postamble();

    const Xbyak::AddressFrame& address_frame(size_t size) const;
    const reg_indices& free_x64regs() const;
    const reg_indices& free_rmmregs() const;

private:
    reg_indices _free_x64regs;
    reg_indices _free_rmmregs;
    internal::consts_table _consts;
    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> _emitters;
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
    variable<float[N]> res(*this);
    uni_vmovups(res, c);
    uni_vfmadd231ps(res, a, b);
    return res;
}

template <size_t N>
jit_kernel::variable<float[N]> jit_kernel::fnma(const variable<float[N]>& a,
                                                 const variable<float[N]>& b,
                                                 const variable<float[N]>& c) {
    variable<float[N]> res(*this);
    uni_vmovups(res, c);
    uni_vfnmadd231ps(res, a, b);
    return res;
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
    auto p = var<T*>();
    p = dst;
    store(p, o0);
    p += step;
    store(p, o1);
    p += step;
    store(p, o2);
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
variable_base<T, register_tag>::variable_base(const variable_base& rhs) : _kernel(rhs._kernel),
                                                                          _reg(rhs._reg) {}

template <typename T>
variable_base<T, register_tag>::variable_base(variable_base&& rhs) noexcept
    : _kernel(rhs._kernel),
      _reg(std::move(rhs._reg)) {}

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

// NOLINTEND(cppcoreguidelines-c-copy-assignment-signature, misc-unconventional-assign-operator)

}  // namespace internal

}  // namespace ov::intel_cpu
