// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cpu/x64/jit_generator.hpp>
#include <emitters/jit_load_store_emitters.hpp>
#include <ie/ie_precision.hpp>
#include <common/nstl.hpp>
#include <type_traits>
#include <functional>
#include <vector>
#include <array>
#include <tuple>

namespace MKLDNNPlugin {

struct jit_kernel;

namespace internal {

template<size_t S>
struct reg_traits_by_size;
template<typename T>
struct reg_traits;
template<typename T, size_t N>
struct reg_traits<T[N]>;
template<dnnl::impl::cpu::x64::cpu_isa_t isa>
struct isa_traits;

template<>
struct reg_traits_by_size<1> {
    using type = Xbyak::Reg8;
    constexpr static size_t size = 1;           // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::isa_any;
};

template<>
struct reg_traits_by_size<2> {
    using type = Xbyak::Reg16;
    constexpr static size_t size = 2;           // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::isa_any;
};

template<>
struct reg_traits_by_size<4> {
    using type = Xbyak::Reg32;
    constexpr static size_t size = 4;           // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::isa_any;
};

template<>
struct reg_traits_by_size<8> {
    using type = Xbyak::Reg64;
    constexpr static size_t size = 8;           // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::isa_any;
};

template<typename T>
struct reg_traits : public reg_traits_by_size<sizeof(T)> {};

template<>
struct reg_traits<float> {
    using type = Xbyak::Fpu;
    constexpr static size_t size = 10;          // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::isa_any;
};
template<>
struct reg_traits<double> : public reg_traits<float> {};

template<typename T>
struct reg_traits<T[1]> : public reg_traits<T> {};

template<typename T>
struct reg_traits<T[4]> {
    using type = Xbyak::Xmm;
    constexpr static size_t size = 4 * 4;       // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::sse41;
};

template<typename T>
struct reg_traits<T[8]> {
    using type = Xbyak::Ymm;
    constexpr static size_t size = 8 * 4;       // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::avx2;
};

template<typename T>
struct reg_traits<T[16]> {
    using type = Xbyak::Zmm;
    constexpr static size_t size = 16 * 4;      // in bytes
    constexpr static dnnl::impl::cpu::x64::cpu_isa_t isa
                        = dnnl::impl::cpu::x64::cpu_isa_t::avx512_common;
};

template<>
struct isa_traits<dnnl::impl::cpu::x64::cpu_isa_t::sse41> {
    struct reg {
        using type = Xbyak::Xmm;
        constexpr static size_t size = 4 * 4;   // in bytes
        constexpr static size_t length = 4;     // in dwords
    };
};

template<>
struct isa_traits<dnnl::impl::cpu::x64::cpu_isa_t::avx2> {
    struct reg {
        using type = Xbyak::Ymm;
        constexpr static size_t size = 8 * 4;   // in bytes
        constexpr static size_t length = 8;     // in dwords
    };
};

template<>
struct isa_traits<dnnl::impl::cpu::x64::cpu_isa_t::avx512_common> {
    struct reg {
        using type = Xbyak::Zmm;
        constexpr static size_t size = 16 * 4;  // in bytes
        constexpr static size_t length = 16;    // in dwords
    };
};

template<typename T>
class variable;
template<typename T>
class if_expression;
template<typename T>
class then_expression;

template<typename T>
class boolean_expression {
private:
    using reg_type = typename reg_traits<T>::type;

    enum class type {
        eq, neq
    };

    boolean_expression(jit_kernel & kernel, type t, const reg_type & lhs, const reg_type & rhs);
    boolean_expression(jit_kernel & kernel, type t, const reg_type & lhs, T rhs);
    void cmp(const Xbyak::Label & exit) const;

    jit_kernel & _kernel;
    type _type;
    const reg_type & _lhs;

    bool _is_ref;

    union datum {
        datum(const reg_type & r)
            : reg(&r) {}
        datum(T v)
            : value(v) {}
        const reg_type * reg;
        T value;
    } _rhs;

    friend class variable<T>;
    friend class if_expression<T>;
    friend class then_expression<T>;
};

template<typename T>
struct then_expression {
    then_expression(if_expression<T> & expr);

    template<typename F>
    void _else(F && fn);

private:
    if_expression<T> & _if_expr;
};

template<typename T>
struct if_expression {
    if_expression(const boolean_expression<T> & expr)
        : _expr(expr) {}

    ~if_expression() {
        try {
            if (!_is_exit_valid)
                _expr._kernel.assignL(_exit, _else);
        } catch(...) {}
    }

    template<typename F>
    then_expression<T> _then(F && fn) {
        using namespace Xbyak;

        _expr.cmp(_else);
        fn();
        _expr._kernel.jmp(_exit, Xbyak::CodeGenerator::T_NEAR);
        _expr._kernel.L(_else);

        return then_expression<T>(*this);
    }

private:
    const boolean_expression<T> & _expr;
    Xbyak::Label _exit;
    Xbyak::Label _else;
    bool _is_exit_valid = false;

    friend class then_expression<T>;
};

template<typename T>
class variable_base {
public:
    using reg_type = typename reg_traits<T>::type;

    variable_base(const variable_base &) = delete;
    variable_base & operator = (const variable_base &) = delete;
    variable_base(variable_base &&);

    operator const reg_type &() const {
        return _reg;
    }

    operator Xbyak::RegExp () const {
        return _reg;
    }

    jit_kernel & kernel;

protected:
    variable_base(jit_kernel & krnl, const reg_type & reg);
    ~variable_base();

    bool _manage_lifetime = true;
    const reg_type & _reg;
};

template<typename T>
class variable : public variable_base<typename std::enable_if<!std::is_floating_point<T>::value, T>::type> {
public:
    using type = T;
    using base = variable_base<type>;
    using reg_type = typename base::reg_type;

    variable(variable &&) = default;
    variable(jit_kernel & krnl);
    variable(jit_kernel & krnl, const reg_type & reg);

    const variable & operator = (const reg_type & rhs) const {
        base::kernel.mov(base::_reg, rhs);
        return *this;
    }
    const variable & operator = (T rhs) const {
        base::kernel.mov(base::_reg, rhs);
        return *this;
    }
    const variable & operator += (const reg_type & rhs) const {
        base::kernel.add(base::_reg, rhs);
        return *this;
    }
    const variable & operator += (typename std::conditional<std::is_pointer<T>::value, size_t, T>::type rhs) const {
        base::kernel.add(base::_reg, rhs);
        return *this;
    }
    const variable & operator -= (const reg_type & rhs) const {
        base::kernel.sub(base::_reg, rhs);
        return *this;
    }
    const variable & operator -= (typename std::conditional<std::is_pointer<T>::value, size_t, T>::type rhs) const {
        base::kernel.sub(base::_reg, rhs);
        return *this;
    }
    const variable & operator &= (const reg_type & rhs) const {
        base::kernel.and_(base::_reg, rhs);
        return *this;
    }
    const variable & operator &= (T rhs) const {
        base::kernel.and_(base::_reg, rhs);
        return *this;
    }
    const variable & operator |= (const reg_type & rhs) const {
        base::kernel.or_(base::_reg, rhs);
        return *this;
    }
    const variable & operator |= (T rhs) const {
        base::kernel.or_(base::_reg, rhs);
        return *this;
    }
    const variable & operator >>= (size_t rhs) const {
        base::kernel.shr(base::_reg, rhs);
        return *this;
    }
    const variable & operator <<= (size_t rhs) const {
        base::kernel.shl(base::_reg, rhs);
        return *this;
    }

    boolean_expression<T> operator == (const reg_type & rhs) const {
        return boolean_expression<T>(base::kernel, boolean_expression<T>::type::eq, base::_reg, rhs);
    }

    boolean_expression<T> operator == (T rhs) const {
        return boolean_expression<T>(base::kernel, boolean_expression<T>::type::eq, base::_reg, rhs);
    }

    boolean_expression<T> operator != (const reg_type & rhs) const {
        return boolean_expression<T>(base::kernel, boolean_expression<T>::type::neq, base::_reg, rhs);
    }

    boolean_expression<T> operator != (T rhs) const {
        return boolean_expression<T>(base::kernel, boolean_expression<T>::type::neq, base::_reg, rhs);
    }

    // TODO: add necessary operations
};

template<typename T, size_t N>
class variable<T[N]> : public variable_base<T[N]> {
public:
    using type = T[N];
    using base = variable_base<type>;
    using reg_type = typename base::reg_type;
    constexpr static size_t length = N;

    variable(variable &&) = default;
    variable(jit_kernel & krnl);
    variable(jit_kernel & krnl, const reg_type & reg);

    const variable & operator = (const reg_type & rhs) const {
        base::kernel.uni_vmovups(base::_reg, rhs);
        return *this;
    }

    const variable & blend(const reg_type & rhs, uint16_t mask) const {
        base::kernel.uni_vblendps(base::_reg, rhs, mask);
        return *this;
    }

    // TODO: implement vector arithmetic
};

class stack_frame {
    stack_frame(const stack_frame &) = delete;
    stack_frame & operator = (const stack_frame &) = delete;

public:
    stack_frame(jit_kernel & kernel, size_t size);
    stack_frame(stack_frame && rhs);
    ~stack_frame();
    const Xbyak::Reg64 & pointer() const;
    void clear() const;

private:
    jit_kernel & _kernel;
    size_t _size;
};

template<typename T>
InferenceEngine::Precision type2precision();

dnnl::impl::cpu::x64::cpu_isa_t get_current_isa();

}   // namespace internal

struct jit_kernel : public dnnl::impl::cpu::x64::jit_generator {
    using reg_indices = std::vector<int>;
    template<typename T>
    using reg_traits = internal::reg_traits<T>;
    template<size_t S>
    using reg_traits_by_size = internal::reg_traits_by_size<S>;
    template<dnnl::impl::cpu::x64::cpu_isa_t isa>
    using isa_traits = internal::isa_traits<isa>;
    using stack_frame = internal::stack_frame;
    template<typename T>
    using variable = internal::variable<T>;
    template<typename T>
    using if_expression = internal::if_expression<T>;
    template<typename T>
    using boolean_expression = internal::boolean_expression<T>;

    template<typename T, typename U>
    Xbyak::Address argPtr(U T::*member) const {
        auto memPtr = &(reinterpret_cast<const T*>(0)->*member);
        const size_t offs =  reinterpret_cast<const char*>(memPtr) - reinterpret_cast<const char*>(0);
        return address_frame(sizeof(U))[param1 + offs];
    }

    template<typename T, typename U>
    variable<U> arg(U T::*member) {
        using traits = internal::reg_traits<U>;
        using reg_type = typename traits::type;
        const auto & res = reserve<reg_type>();
        if (sizeof(T) < traits::size)
            movzx(res, argPtr(member));
        else
            mov(res, argPtr(member));
        return { *this, res };
    }

    template<typename CastU, typename T, typename U>
    variable<CastU> arg(U T::*member) {
        using traits = internal::reg_traits<U>;
        using reg_type = typename traits::type;
        const auto & res = reserve<reg_type>();
        if (sizeof(T) < traits::size)
            movzx(res, argPtr(member));
        else
            mov(res, argPtr(member));
        return { *this, res };
    }

    jit_kernel();

    template<typename RegType>
    const RegType & reserve();

    template<typename RegType>
    void free(const RegType & reg);

    template<typename T>
    void copy(const Xbyak::Reg64& dst,
              const Xbyak::Reg64& src,
              const Xbyak::Reg64& size);
    template<typename T>
    void copy(const Xbyak::Address& dst,
              const Xbyak::Reg64& src,
              const Xbyak::Reg64& size);

    template<typename DstT, typename SrcT>
    void load(const variable<DstT> & dst, const variable<SrcT> & src);
    template<typename DstT, typename SrcT>
    void store(const variable<DstT> & dst, const variable<SrcT> & src);

    template<typename B, typename E, typename S = size_t>
    void foreach(const B & begin,
                 const E & end,
                 std::function<void(const Xbyak::Reg64&)> && fn,
                 const S & step = 1);

    template<typename T>
    variable<T> var();

    stack_frame stack(size_t size);

    template<typename T>
    if_expression<T> _if(const boolean_expression<T> & expr);

    void uni_vpermps(const Xbyak::Xmm& x1, const int *mask, const Xbyak::Operand& op);
    void uni_vpermps(const Xbyak::Ymm& y1, const int *mask, const Xbyak::Operand& op);
    void uni_vpermps(const Xbyak::Zmm& z1, const int *mask, const Xbyak::Operand& op);
    void uni_vblendps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, uint16_t mask);
    void uni_vblendps(const Xbyak::Ymm& y1, const Xbyak::Ymm& y2, uint16_t mask);
    void uni_vblendps(const Xbyak::Zmm& z1, const Xbyak::Zmm& z2, uint16_t mask);

    void postamble();

private:
    const Xbyak::AddressFrame & address_frame(size_t size) const;

    reg_indices _free_x64regs;
    reg_indices _free_rmmregs;
    bool _is_load_emitter_used = false;
    bool _is_store_emitter_used = false;
    jit_load_emitter _load_emitter;
    jit_store_emitter _store_emitter;
};

template<typename T>
void jit_kernel::copy(const Xbyak::Reg64& dst,
                      const Xbyak::Reg64& src,
                      const Xbyak::Reg64& size) {
    const auto & addr_frame = address_frame(sizeof(T));
    auto p = reserve<typename reg_traits_by_size<sizeof(T)>::type>();
    foreach(0, size, [&](const Xbyak::Reg64& idx) {
        mov(p, addr_frame[src + idx * sizeof(T)]);
        mov(addr_frame[dst + idx * sizeof(T)], p);
    });
    free(p);
}

template<typename T>
void jit_kernel::copy(const Xbyak::Address& dst,
                      const Xbyak::Reg64& src,
                      const Xbyak::Reg64& size) {
    const auto & addr_frame = address_frame(sizeof(T));
    auto p = reserve<typename reg_traits_by_size<sizeof(T)>::type>();
    auto d = reserve<Xbyak::Reg64>();
    lea(d, dst);
    foreach(0, size, [&](const Xbyak::Reg64& idx) {
        mov(p, addr_frame[src + idx * sizeof(T)]);
        mov(addr_frame[d + idx * sizeof(T)], p);
    });
    free(d);
    free(p);
}

template<typename DstT, typename SrcT>
void jit_kernel::load(const variable<DstT> & dst, const variable<SrcT> & src) {
    static_assert(std::is_same<typename variable<SrcT>::reg_type, Xbyak::Reg64>::value,
        "Source register must be Reg64");

    using src_type = typename std::remove_cv<
                        typename std::remove_pointer<
                            typename std::decay<SrcT>::type>::type>::type;
    using dst_type = typename std::remove_cv<
                        typename std::remove_pointer<
                            typename std::decay<DstT>::type>::type>::type;
    constexpr size_t length = variable<DstT>::length;

    const std::vector<size_t> pool_vec_idxs(_free_rmmregs.begin(), _free_rmmregs.end());
    const std::vector<size_t> pool_gpr_idxs(_free_x64regs.begin(), _free_x64regs.end());

    _load_emitter.emit_code(
        { static_cast<size_t>(static_cast<const Xbyak::Operand&>(src).getIdx()) },
        { static_cast<size_t>(static_cast<const Xbyak::Operand&>(dst).getIdx()) },
        std::make_shared<load_emitter_context>(
            internal::type2precision<src_type>(),
            internal::type2precision<dst_type>(),
            static_cast<int>(length)),
        pool_vec_idxs,
        pool_gpr_idxs);

    _is_load_emitter_used = true;
}

template<typename DstT, typename SrcT>
void jit_kernel::store(const variable<DstT> & dst, const variable<SrcT> & src) {
    static_assert(std::is_same<typename variable<DstT>::reg_type, Xbyak::Reg64>::value,
        "Destibnation register must be Reg64");

    using src_type = typename std::remove_cv<
                        typename std::remove_pointer<
                            typename std::decay<SrcT>::type>::type>::type;
    using dst_type = typename std::remove_cv<
                        typename std::remove_pointer<
                            typename std::decay<DstT>::type>::type>::type;
    constexpr size_t length = variable<SrcT>::length;

    const std::vector<size_t> pool_vec_idxs(_free_rmmregs.begin(), _free_rmmregs.end());
    const std::vector<size_t> pool_gpr_idxs(_free_x64regs.begin(), _free_x64regs.end());

    _store_emitter.emit_code(
        { static_cast<size_t>(static_cast<const Xbyak::Operand&>(src).getIdx()) },
        { static_cast<size_t>(static_cast<const Xbyak::Operand&>(dst).getIdx()) },
        std::make_shared<store_emitter_context>(
            internal::type2precision<src_type>(),
            internal::type2precision<dst_type>(),
            static_cast<int>(length)),
        pool_vec_idxs,
        pool_gpr_idxs);

    _is_store_emitter_used = true;
}

template<typename B, typename E, typename S>
void jit_kernel::foreach(const B & begin,
                         const E & end,
                         std::function<void(const Xbyak::Reg64&)> && fn,
                         const S & step) {
    using namespace Xbyak;

    Label loop, exit;

    auto idx = reserve<Reg64>();

    mov(idx, begin);

    L(loop);
    cmp(idx, end);
    jge(exit, T_NEAR);

    fn(idx);

    add(idx, step);
    jmp(loop, T_NEAR);
    L(exit);

    free<Reg64>(idx);
}

template<typename T>
jit_kernel::variable<T> jit_kernel::var() {
    using reg_type = typename reg_traits<T>::type;
    const auto & reg = reserve<reg_type>();
    return variable<T>(*this, reg);
}

template<typename T>
jit_kernel::if_expression<T> jit_kernel::_if(const boolean_expression<T> & expr) {
    return if_expression<T>(expr);
}

namespace internal {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// boolean_expression

template<typename T>
boolean_expression<T>::boolean_expression(jit_kernel & kernel, type t, const reg_type & lhs, const reg_type & rhs)
    : _kernel(kernel)
    , _type(t)
    , _lhs(lhs)
    , _is_ref(true)
    , _rhs(rhs) {
}

template<typename T>
boolean_expression<T>::boolean_expression(jit_kernel & kernel, type t, const reg_type & lhs, T rhs)
    : _kernel(kernel)
    , _type(t)
    , _lhs(lhs)
    , _is_ref(false)
    , _rhs(rhs) {
}

template<typename T>
void boolean_expression<T>::cmp(const Xbyak::Label & exit) const {
    if (_is_ref)
        _kernel.cmp(_lhs, *_rhs.reg);
    else
        _kernel.cmp(_lhs, _rhs.value);

    switch (_type) {
        case type::eq: {
            _kernel.jne(exit, Xbyak::CodeGenerator::T_NEAR);
            break;
        }
        case type::neq: {
            _kernel.je(exit, Xbyak::CodeGenerator::T_NEAR);
            break;
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// then_expression

template<typename T>
then_expression<T>::then_expression(if_expression<T> & expr)
    : _if_expr(expr) {}

template<typename T>
template<typename F>
void then_expression<T>::_else(F && fn) {
    fn();
    _if_expr._expr._kernel.L(_if_expr._exit);
    _if_expr._is_exit_valid = true;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// variable

template<typename T>
variable_base<T>::variable_base(jit_kernel & krnl, const reg_type & reg)
    : kernel(krnl)
    , _reg(reg) {
}

template<typename T>
variable_base<T>::variable_base(variable_base && rhs)
    : kernel(rhs.kernel)
    , _reg(rhs._reg) {
    rhs._manage_lifetime = false;
}

template<typename T>
variable_base<T>::~variable_base() {
    if (_manage_lifetime)
        kernel.free(_reg);
}

template<typename T>
variable<T>::variable(jit_kernel & krnl)
    : base(krnl, krnl.reserve<typename reg_traits<T>::type>()) {
}

template<typename T>
variable<T>::variable(jit_kernel & krnl, const reg_type & reg)
    : base(krnl, reg) {
}

template<typename T, size_t N>
variable<T[N]>::variable(jit_kernel & krnl)
    : base(krnl, krnl.reserve<typename reg_traits<T[N]>::type>()) {
}

template<typename T, size_t N>
variable<T[N]>::variable(jit_kernel & krnl, const reg_type & reg)
    : base(krnl, reg) {
}

}   // namespace internal

}   // namespace MKLDNNPlugin
