/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_JIT_CONV_IR_CORE_HPP
#define GPU_JIT_CONV_IR_CORE_HPP

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <numeric>
#include <string>

#include "common/c_types_map.hpp"
#include "gpu/jit/conv/ngen_proxy.hpp"
#include "gpu/jit/conv/utils.hpp"

// All IR expression objects.
#define HANDLE_EXPR_IR_OBJECTS() \
    HANDLE_IR_OBJECT(binary_op_t) \
    HANDLE_IR_OBJECT(bool_imm_t) \
    HANDLE_IR_OBJECT(cast_t) \
    HANDLE_IR_OBJECT(float_imm_t) \
    HANDLE_IR_OBJECT(iif_t) \
    HANDLE_IR_OBJECT(int_imm_t) \
    HANDLE_IR_OBJECT(load_t) \
    HANDLE_IR_OBJECT(ptr_t) \
    HANDLE_IR_OBJECT(shuffle_t) \
    HANDLE_IR_OBJECT(ternary_op_t) \
    HANDLE_IR_OBJECT(unary_op_t) \
    HANDLE_IR_OBJECT(var_t)

// All IR statement objects.
#define HANDLE_STMT_IR_OBJECTS() \
    HANDLE_IR_OBJECT(alloc_t) \
    HANDLE_IR_OBJECT(for_t) \
    HANDLE_IR_OBJECT(func_call_t) \
    HANDLE_IR_OBJECT(if_t) \
    HANDLE_IR_OBJECT(let_t) \
    HANDLE_IR_OBJECT(stmt_group_t) \
    HANDLE_IR_OBJECT(stmt_seq_t) \
    HANDLE_IR_OBJECT(store_t)

#define HANDLE_ALL_IR_OBJECTS() \
    HANDLE_EXPR_IR_OBJECTS() \
    HANDLE_STMT_IR_OBJECTS() \
    HANDLE_IR_OBJECT(func_impl_t)

enum ir_type_id_t {
#define HANDLE_IR_OBJECT(type) type,

    // Create typeid for objects which can be visited/mutated. These need to be
    // first as the typeid is used as an index into an array to dispatch to the
    // correct mutate function.
    HANDLE_ALL_IR_OBJECTS()

    //Used to calculate number of IR objects that can be visited/mutated
    end_visitable_ir_objects,

    // Other IR object
    expr_impl_t = end_visitable_ir_objects,
    nary_op_t,
    stmt_impl_t,
    grf_alloc_attr_t,
    instruction_modifier_attr_t,
    builtin_t,
    pexpr_t,
    pint_imm_t,
    factored_expr_t,
    send_t,
    dpas_t,
    mad_t,
    reduce_t,
    reorder_t,
    eltwise_t,

#undef HANDLE_IR_OBJECT
};

// Auxiliary macros to reduce boilerplate.
#define IR_DECL_TYPE_ID(class_name) \
    using self_type = class_name; \
    static int64_t _type_id() { return ir_type_id_t::class_name; } \
    static int64_t _dispatch_type_id() { return _type_id(); } \
    int64_t type_id() const override { return _type_id(); }

#define IR_DECL_DERIVED_TYPE_ID(class_name, base_name) \
    using self_type = class_name; \
    static int64_t _type_id() { return ir_type_id_t::class_name; } \
    int64_t type_id() const override { return _type_id(); } \
    static int64_t _dispatch_type_id() { return base_name::_type_id(); } \
    int64_t dispatch_type_id() const override { return _dispatch_type_id(); }

#define IR_DECL_EXPR_TYPE_ID(class_name) \
    IR_DECL_TYPE_ID(class_name) \
    bool is_expr() const override { return true; }

#define IR_DECL_STMT_TYPE_ID(class_name) \
    IR_DECL_TYPE_ID(class_name) \
    bool is_stmt() const override { return true; }

// Defines getter for a function argument.
#define IR_DEFINE_ARG_GET(name, index) \
    static const expr_t &arg_##name(const stmt_t &s) { \
        ir_assert(s.is<func_call_t>()) << s; \
        auto &c = s.as<func_call_t>(); \
        ir_assert(c.func.is<self_type>()) << s; \
        return c.args[index]; \
    } \
    template <typename T> \
    static T &arg_##name(std::vector<T> &args) { \
        return args[index]; \
    } \
    template <typename T> \
    static const T &arg_##name(const std::vector<T> &args) { \
        return args[index]; \
    }

#if defined(__GNUC__)
// clang-format off
// Defines dump() method for debugging purposes, to pretty print the object.
#define IR_DEFINE_DUMP() \
    __attribute__((noinline)) \
    __attribute__((used)) \
    void dump() const { \
        printf("%s\n", str().c_str()); \
    }
// clang-format on
#else
#define IR_DEFINE_DUMP()
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

enum class type_kind_t {
    undef,
    _bool,

    // Integer types.
    u8,
    s8,
    u16,
    s16,
    u32,
    s32,
    u64,
    s64,

    // Floating point types.
    bf16,
    f16,
    f32,

    // Message data types.
    byte,
    dword,
    qword,
    oword,
    hword
};

std::string to_string(type_kind_t kind);

class type_t {
public:
    static type_t undef() { return type_t(type_kind_t::undef); }
    static type_t _bool(int elems = 1) {
        return type_t(type_kind_t::_bool, elems);
    }

    static type_t u8(int elems = 1) { return type_t(type_kind_t::u8, elems); }
    static type_t s8(int elems = 1) { return type_t(type_kind_t::s8, elems); }
    static type_t u16(int elems = 1) { return type_t(type_kind_t::u16, elems); }
    static type_t s16(int elems = 1) { return type_t(type_kind_t::s16, elems); }
    static type_t u32(int elems = 1) { return type_t(type_kind_t::u32, elems); }
    static type_t s32(int elems = 1) { return type_t(type_kind_t::s32, elems); }
    static type_t u64(int elems = 1) { return type_t(type_kind_t::u64, elems); }
    static type_t s64(int elems = 1) { return type_t(type_kind_t::s64, elems); }

    // Returns unsigned integer type.
    static type_t u(int bits, int elems = 1) {
        switch (bits) {
            case 8: return u8(elems);
            case 16: return u16(elems);
            case 32: return u32(elems);
            case 64: return u64(elems);
            default: ir_error_not_expected();
        }
        return type_t::undef();
    }

    // Returns signed integer type.
    static type_t s(int bits, int elems = 1) {
        switch (bits) {
            case 8: return s8(elems);
            case 16: return s16(elems);
            case 32: return s32(elems);
            case 64: return s64(elems);
            default: ir_error_not_expected();
        }
        return type_t::undef();
    }

    static type_t bf16(int elems = 1) {
        return type_t(type_kind_t::bf16, elems);
    }
    static type_t f16(int elems = 1) { return type_t(type_kind_t::f16, elems); }
    static type_t f32(int elems = 1) { return type_t(type_kind_t::f32, elems); }

    static type_t byte(int elems = 1) {
        return type_t(type_kind_t::byte, elems);
    }
    static type_t byte_ptr(int elems = 1) {
        return type_t(type_kind_t::byte, elems).with_ptr();
    }
    static type_t dword(int elems = 1) {
        return type_t(type_kind_t::dword, elems);
    }
    static type_t qword(int elems = 1) {
        return type_t(type_kind_t::qword, elems);
    }
    static type_t oword(int elems = 1) {
        return type_t(type_kind_t::oword, elems);
    }
    static type_t hword(int elems = 1) {
        return type_t(type_kind_t::hword, elems);
    }

    template <typename T>
    static type_t from_cpp() {
#define CASE(cpp_type, type) \
    if (std::is_same<T, cpp_type>::value) return type()

        CASE(bool, _bool);
        CASE(float, f32);
        CASE(int16_t, s16);
        CASE(int32_t, s32);
        CASE(int64_t, s64);
        CASE(uint16_t, u16);
        CASE(uint32_t, u32);
        CASE(uint64_t, u64);

#undef CASE

        ir_error_not_expected();

        return undef();
    }

    static bool is_vector(int elems) { return elems != 1; }

    type_t() : type_t(type_t::undef()) {}

    type_t(type_kind_t kind, uint32_t elems = 1) : kind_(kind), elems_(elems) {}

    // Constructor from dnnl_data_type_t.
    type_t(data_type_t dt) {
        elems_ = 1;
        switch (dt) {
#define CASE(x) \
    case data_type::x: kind_ = type_kind_t::x; break;
            CASE(bf16);
            CASE(f16);
            CASE(f32);
            CASE(s32);
            CASE(s8);
            CASE(u8);
#undef CASE
            default: ir_error_not_expected();
        }
    }

    type_kind_t kind() const { return kind_; }

    int elems() const { return elems_; }

    bool is_ptr() const { return is_ptr_; }

    bool operator==(const type_t &other) const {
        return (kind() == other.kind()) && (elems() == other.elems())
                && (is_ptr() == other.is_ptr());
    }

    bool operator!=(const type_t &other) const { return !operator==(other); }

    bool is_equal(const type_t &other) const { return operator==(other); }

    size_t get_hash() const {
        return ir_utils::get_hash(kind(), elems(), is_ptr());
    }

    bool is_undef() const { return kind() == type_kind_t::undef; }

    bool is_vector() const { return type_t::is_vector(elems()); }

    bool is_bool() const { return kind() == type_kind_t::_bool; }

    bool is_fp() const {
        return utils::one_of(
                kind(), type_kind_t::bf16, type_kind_t::f16, type_kind_t::f32);
    }

    bool is_bf16() const { return kind() == type_kind_t::bf16; }
    bool is_f16() const { return kind() == type_kind_t::f16; }
    bool is_f32() const { return kind() == type_kind_t::f32; }

    bool is_int() const {
        return utils::one_of(kind(), type_kind_t::u8, type_kind_t::s8,
                type_kind_t::u16, type_kind_t::s16, type_kind_t::u32,
                type_kind_t::s32, type_kind_t::u64, type_kind_t::s64);
    }

    bool is_s8() const { return kind() == type_kind_t::s8; }
    bool is_u8() const { return kind() == type_kind_t::u8; }
    bool is_x8() const {
        return utils::one_of(kind(), type_kind_t::s8, type_kind_t::u8);
    }

    bool is_s16() const { return kind() == type_kind_t::s16; }
    bool is_u16() const { return kind() == type_kind_t::u16; }
    bool is_x16() const {
        return utils::one_of(kind(), type_kind_t::s16, type_kind_t::u16);
    }

    bool is_s32() const { return kind() == type_kind_t::s32; }
    bool is_u32() const { return kind() == type_kind_t::u32; }
    bool is_x32() const {
        return utils::one_of(kind(), type_kind_t::s32, type_kind_t::u32);
    }

    bool is_signed(int elems = -1) const {
        if (elems != -1 && elems_ != elems) return false;
        return utils::one_of(kind(), type_kind_t::s8, type_kind_t::s16,
                type_kind_t::s32, type_kind_t::s64);
    }

    bool is_unsigned(int elems = -1) const {
        if (elems != -1 && elems_ != elems) return false;
        return utils::one_of(kind(), type_kind_t::u8, type_kind_t::u16,
                type_kind_t::u32, type_kind_t::u64);
    }

    bool is_scalar() const { return elems() == 1; }

    template <typename T>
    bool is_cpp() const {
        return *this == type_t::from_cpp<T>();
    }

    type_t remove_elems() const { return with_elems(1); }

    type_t remove_ptr() const {
        type_t copy = *this;
        copy.is_ptr_ = false;
        return copy;
    }

    type_t with_elems(int new_elems) const {
        type_t copy = *this;
        copy.elems_ = new_elems;
        return copy;
    }

    type_t with_ptr() const {
        type_t copy = *this;
        copy.is_ptr_ = true;
        return copy;
    }

    type_t scalar() const { return with_elems(1); }

    // Returns size in bytes.
    int size() const;

    std::string str() const {
        std::ostringstream oss;
        oss << to_string(kind());
        if (elems() > 1) oss << "x" << elems();
        if (is_ptr()) oss << "*";
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    type_kind_t kind_ = type_kind_t::undef;
    int elems_ = 0;
    bool is_ptr_ = false;
};

inline std::ostream &operator<<(std::ostream &out, const type_t &type) {
    out << type.str();
    return out;
}

// type_t to dnnl_data_type_t convertor.
data_type_t to_dnnl(const type_t &type);

// Reference counter for IR objects.
class ref_count_t {
public:
    ref_count_t() : value_(0) {}
    ref_count_t(const ref_count_t &) = delete;

    uint32_t increment() { return ++value_; }
    uint32_t decrement() { return --value_; }

private:
    std::atomic<uint32_t> value_;
};

// Base class for all IR objects. Implemented as an intrusive pointer, with
// the reference counter stored inside the object.
class object_impl_t {
public:
    object_impl_t() = default;

    object_impl_t(const object_impl_t &) = delete;

    virtual ~object_impl_t() = default;

    ref_count_t &ref_count() { return ref_count_; }

    // Unique type ID.
    virtual int64_t type_id() const = 0;

    // Type ID used for dispatching in ir_visitor_t and ir_mutator_t.
    // For some IR objects
    virtual int64_t dispatch_type_id() const { return type_id(); }

    // Provides equality semantics.
    virtual bool is_equal(const object_impl_t *obj) const = 0;

    virtual size_t get_hash() const = 0;

    virtual bool is_expr() const { return false; }
    virtual bool is_stmt() const { return false; }

    // Downcasts the object to the IR type, returns a reference. The IR type
    // must match the real IR type.
    template <typename T>
    const T &as() const {
        ir_assert(type_id() == T::_type_id());
        return *(const T *)this;
    }

    template <typename T>
    T &as() {
        ir_assert(type_id() == T::_type_id());
        return *(T *)this;
    }

    // Downcasts the object to the IR type, returns a pointer. If the IR type
    // doesn't match the real IR type, returns nullptr.
    template <typename T>
    const T *as_ptr() const {
        if (type_id() != T::_type_id()) return nullptr;
        return (const T *)this;
    }

    template <typename T>
    T *as_ptr() {
        if (type_id() != T::_type_id()) return nullptr;
        return (T *)this;
    }

    // Returns true if T matches the real IR type.
    template <typename T>
    bool is() const {
        return type_id() == T::_type_id();
    }

    virtual std::string str() const;

    IR_DEFINE_DUMP()

private:
    ref_count_t ref_count_;
};

// Base wrapper for IR objects.
class object_t {
public:
    object_t(object_impl_t *impl = nullptr) : impl_(impl) {
        increment(impl_);
#ifndef NDEBUG
        sanity_check();
#endif
    }
    object_t(const object_impl_t *impl)
        : object_t(const_cast<object_impl_t *>(impl)) {}
    object_t(const object_t &obj) : object_t(obj.impl()) {}
    object_t(object_t &&obj) : impl_(obj.impl_) {
        obj.impl_ = nullptr;
#ifndef NDEBUG
        sanity_check();
#endif
    }

    virtual ~object_t() { decrement_and_maybe_destroy(impl_); }

    object_t &operator=(const object_t &other) {
        increment(other.impl());
        decrement_and_maybe_destroy(impl_);
        impl_ = other.impl();
#ifndef NDEBUG
        sanity_check();
#endif
        return *this;
    }

    object_t &operator=(object_t &&other) {
        std::swap(impl_, other.impl_);
#ifndef NDEBUG
        sanity_check();
#endif
        return *this;
    }

    object_impl_t *impl() const { return impl_; }

    bool is_empty() const { return !impl_; }

    int64_t type_id() const { return impl_->type_id(); }

    int64_t dispatch_type_id() const { return impl_->dispatch_type_id(); }

    template <typename T>
    const T &as() const {
        ir_assert(impl_);
        return impl_->as<T>();
    }

    template <typename T>
    T &as() {
        ir_assert(impl_);
        return impl_->as<T>();
    }

    template <typename T>
    const T *as_ptr() const {
        if (!impl_) return nullptr;
        return impl_->as_ptr<T>();
    }

    template <typename T>
    T *as_ptr() {
        if (!impl_) return nullptr;
        return impl_->as_ptr<T>();
    }

    template <typename T>
    bool is() const {
        if (is_empty()) return false;
        return impl_->is<T>();
    }

    // Comparison with identity semantics.
    bool is_same(const object_t &other) const { return impl_ == other.impl(); }

    // Comparison with equality semantics.
    bool is_equal(const object_t &other) const {
        if (is_empty() || other.is_empty())
            return is_empty() == other.is_empty();

        return impl_->is_equal(other.impl());
    }

    size_t get_hash() const {
        if (is_empty()) return 0;
        return impl()->get_hash();
    }

    bool is_expr() const { return impl_ && impl_->is_expr(); }
    bool is_stmt() const { return impl_ && impl_->is_stmt(); }

    std::string str() const {
        if (is_empty()) return "(nil)";
        return impl()->str();
    }

    IR_DEFINE_DUMP()

protected:
    virtual void sanity_check() const {}

private:
    static void increment(object_impl_t *impl) {
        if (!impl) return;
        impl->ref_count().increment();
    }

    static void decrement_and_maybe_destroy(object_impl_t *impl) {
        if (!impl) return;
        if (impl->ref_count().decrement() == 0) { delete impl; }
    }

    object_impl_t *impl_;
};

inline std::ostream &operator<<(std::ostream &out, const object_t &obj) {
    out << obj.str();
    return out;
}

// Helper classes for containers to store object_t.
struct object_id_hash_t {
    size_t operator()(const object_t &obj) const {
        return std::hash<const object_impl_t *>()(obj.impl());
    }
};

struct object_eq_hash_t {
    size_t operator()(const object_t &obj) const { return obj.get_hash(); }
};

struct object_id_equal_t {
    bool operator()(const object_t &a, const object_t &b) const {
        return a.is_same(b);
    }
};

struct object_eq_equal_t {
    bool operator()(const object_t &a, const object_t &b) const {
        return a.is_equal(b);
    }
};

// Containers to store object_t.

// Unordered set, uses identity comparison for keys.
template <typename KeyT>
using object_set_t
        = std::unordered_set<KeyT, object_id_hash_t, object_id_equal_t>;

// Unordered set, uses equality comparison for keys.
template <typename KeyT>
using object_eq_set_t
        = std::unordered_set<KeyT, object_eq_hash_t, object_eq_equal_t>;

// Unordered map, uses identity comparison for keys.
template <typename KeyT, typename ValueT>
using object_map_t
        = std::unordered_map<KeyT, ValueT, object_id_hash_t, object_id_equal_t>;

// Unordered map, uses equality comparison for keys.
template <typename KeyT, typename ValueT>
using object_eq_map_t
        = std::unordered_map<KeyT, ValueT, object_eq_hash_t, object_eq_equal_t>;

// Base class for IR expression objects.
class expr_impl_t : public object_impl_t {
public:
    IR_DECL_TYPE_ID(expr_impl_t)

    expr_impl_t(const type_t &type) : type(type) {}

    type_t type;
};

// Wrapper for IR expression objects.
class expr_t : public object_t {
public:
    using object_t::object_t;

    expr_t() = default;
    expr_t(const object_t &obj) : object_t(obj) {}
    expr_t(object_t &&obj) : object_t(obj) {}
    expr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    expr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    explicit expr_t(bool v);
    expr_t(float v);
    expr_t(int16_t v);
    expr_t(int32_t v);
    expr_t(int64_t v);
    expr_t(uint16_t v);
    expr_t(uint32_t v);
    expr_t(uint64_t v);

    const type_t &type() const {
        ir_assert(!is_empty());
        return ((const expr_impl_t *)impl())->type;
    }

#define DECLARE_BINARY_ASSIGN_OPERATOR(op) \
    expr_t &operator op##=(const expr_t &rhs);

    DECLARE_BINARY_ASSIGN_OPERATOR(+)
    DECLARE_BINARY_ASSIGN_OPERATOR(-)
    DECLARE_BINARY_ASSIGN_OPERATOR(*)
    DECLARE_BINARY_ASSIGN_OPERATOR(/)
    DECLARE_BINARY_ASSIGN_OPERATOR(%)
    DECLARE_BINARY_ASSIGN_OPERATOR(&)

#undef DECLARE_BINARY_ASSIGN_OPERATOR

    // Returns a pointer shifted by `off` bytes relative to this pointer. The
    // base expression must be a pointer.
    expr_t operator[](const expr_t &off) const;

private:
    void sanity_check() const override {
        ir_assert(dynamic_cast<const expr_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
};

// Helper functions.
inline bool is_const(const expr_t &e);
inline bool is_var(const expr_t &e);

// Unary and binary operators.
enum class op_kind_t {
    undef,

    _minus,
    _add,
    _sub,
    _mul,
    _div,
    _mod,
    _shl,
    _shr,
    _min,
    _max,

    _lt,
    _le,
    _gt,
    _ge,
    _ne,
    _eq,

    _and,

    _add3, // a + b + c
    _mad, // a + b * c
};

std::string to_string(op_kind_t kind);

inline std::ostream &operator<<(std::ostream &out, op_kind_t kind) {
    out << to_string(kind);
    return out;
}

bool is_cmp_op(op_kind_t op_kind);

op_kind_t negate_cmp_op(op_kind_t op_kind);

type_t unary_op_type(op_kind_t op_kind, const expr_t &a);

type_t common_int_type(const type_t &_a, const type_t &_b);

type_t common_type(const type_t &a, const type_t &b);

type_t common_type(const expr_t &a, const expr_t &b);

type_t binary_op_type(op_kind_t op_kind, const expr_t &a, const expr_t &b);

type_t ternary_op_type(
        op_kind_t op_kind, const expr_t &a, const expr_t &b, const expr_t &c);

type_t nary_op_type(op_kind_t op_kind, const std::vector<expr_t> &args);

// Binary operation: (a op b).
class binary_op_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(binary_op_t)

    static expr_t make(op_kind_t op_kind, const expr_t &a, const expr_t &b) {
        return expr_t(new binary_op_t(op_kind, a, b));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (op_kind == other.op_kind) && a.is_equal(other.a)
                && b.is_equal(other.b);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(op_kind, a, b);
    }

    op_kind_t op_kind;
    expr_t a;
    expr_t b;

private:
    binary_op_t(op_kind_t op_kind, const expr_t &a, const expr_t &b)
        : expr_impl_t(binary_op_type(op_kind, a, b))
        , op_kind(op_kind)
        , a(a)
        , b(b) {}
};

// Boolean immediate value.
class bool_imm_t : public expr_impl_t {
public:
    friend class expr_t;
    IR_DECL_EXPR_TYPE_ID(bool_imm_t)

    static expr_t make(bool value) { return expr_t(new bool_imm_t(value)); }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return value == other.value;
    }

    size_t get_hash() const override { return ir_utils::get_hash(value); }

    bool value;

private:
    bool_imm_t(bool value) : expr_impl_t(type_t::_bool()), value(value) {}
};

// Cast between data types.
class cast_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(cast_t)

    static expr_t make(
            const type_t &type, const expr_t &expr, bool saturate = false) {
        return expr_t(new cast_t(type, expr, saturate));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (type == other.type) && expr.is_equal(other.expr)
                && (saturate == other.saturate);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(type, expr, saturate);
    }

    expr_t expr;
    bool saturate;

private:
    cast_t(const type_t &type, const expr_t &expr, bool saturate)
        : expr_impl_t(type), expr(expr), saturate(saturate) {
        ir_assert(type.elems() == expr.type().elems())
                << "Number of elements must match.";
    }
};

// Floating-point immediate value.
class float_imm_t : public expr_impl_t {
public:
    friend class expr_t;
    IR_DECL_EXPR_TYPE_ID(float_imm_t)

    static expr_t make(float value) { return expr_t(new float_imm_t(value)); }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return value == other.value;
    }

    size_t get_hash() const override { return ir_utils::get_hash(value); }

    float value;

private:
    float_imm_t(float value) : expr_impl_t(type_t::f32()), value(value) {}
};

// Immediate if or the conditional (ternary) operator.
// C++ equivalent: (cond ? true_expr : false_expr).
class iif_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(iif_t);

    static expr_t make(const expr_t &cond, const expr_t &true_expr,
            const expr_t &false_expr) {
        return expr_t(new iif_t(cond, true_expr, false_expr));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return cond.is_equal(other.cond) && true_expr.is_equal(other.true_expr)
                && false_expr.is_equal(other.false_expr);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(cond, true_expr, false_expr);
    }

    expr_t cond;
    expr_t true_expr;
    expr_t false_expr;

private:
    iif_t(const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr)
        : expr_impl_t(common_type(true_expr.type(), false_expr.type()))
        , cond(cond)
        , true_expr(true_expr)
        , false_expr(false_expr) {}
};

// Integer immediate value.
class int_imm_t : public expr_impl_t {
public:
    friend class expr_t;
    IR_DECL_EXPR_TYPE_ID(int_imm_t);

    template <typename T>
    static expr_t make(T value, const type_t &type = type_t::undef()) {
        return expr_t(new int_imm_t(value, type));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return value == other.value;
    }

    size_t get_hash() const override { return ir_utils::get_hash(value); }

    static expr_t shrink_type(const expr_t &e) {
        auto &imm = e.as<int_imm_t>();
        type_t new_type = shrink_type(imm.value);
        if (new_type == imm.type) return e;
        return make(imm.value, new_type);
    }

    template <typename T>
    static bool try_shrink_type(int64_t v) {
        if (v >= std::numeric_limits<T>::min()
                && v <= std::numeric_limits<T>::max())
            return true;
        return false;
    }

    int64_t value;

private:
    int_imm_t(int64_t value, const type_t &type = type_t::undef())
        : expr_impl_t(type.is_undef() ? shrink_type(value) : type)
        , value(value) {}

    static type_t shrink_type(int64_t v) {
        if (try_shrink_type<int32_t>(v)) return type_t::s32();
        return type_t::s64();
    }
};

// Updates `base_expr` and `off` so that after return:
// - base_expr contains a variable of a pointer type
// - off contains an offset
void normalize_ptr(const type_t &type, expr_t &base, expr_t &off);

// Load from a GRF buffer.
// C++ equivalent (when type is scalar):
//     load = *(type *)(&buf[off]);
// C++ equivalent (when type is vector):
//     int _stride = (has_default_stride() ? sizeof(scalar_type) : stride);
//     for (int i = 0; i < elems; i++) {
//         load[i] = *(scalar_type *)(&buf[off + i * _stride]);
//     }
class load_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(load_t)

    // offset and stride are expressed in bytes.
    // default stride means unit stride (in terms of value.type().scalar()
    // elements).
    static expr_t make(const type_t &type, const expr_t &buf, const expr_t &off,
            int stride = default_stride) {
        return expr_t(new load_t(type, buf, off, stride));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return type.is_equal(other.type) && buf.is_equal(other.buf)
                && off.is_equal(other.off) && (stride == other.stride);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(type, buf, off, stride);
    }

    bool has_default_stride() const { return stride == default_stride; }

    static const int default_stride = -1;

    expr_t buf;
    expr_t off;
    int stride;

private:
    load_t(const type_t &_type, const expr_t &_buf, const expr_t &_off,
            int _stride)
        : expr_impl_t(_type), buf(_buf), off(_off), stride(_stride) {
        normalize_ptr(type, buf, off);
        ir_assert(is_var(buf)) << buf;
        ir_assert(buf.type().is_ptr()) << buf;
        if (stride == type.scalar().size()) stride = default_stride;
    }
};

// N-ary expression: (a[0] op a[1] op ... op a[n - 1]),
// where <op> is either addition or multiplication.
class nary_op_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(nary_op_t)

    static expr_t make(op_kind_t op_kind, const std::vector<expr_t> &args) {
        return expr_t(new nary_op_t(op_kind, args));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (op_kind == other.op_kind)
                && ir_utils::is_equal(args, other.args);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(op_kind, args);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << "(";
        for (size_t i = 0; i < args.size(); i++) {
            oss << (i != 0 ? " " + to_string(op_kind) + " " : "") << args[i];
        }

        oss << ")";
        return oss.str();
    }

    op_kind_t op_kind;
    std::vector<expr_t> args;

private:
    nary_op_t(op_kind_t op_kind, const std::vector<expr_t> &args)
        : expr_impl_t(nary_op_type(op_kind, args))
        , op_kind(op_kind)
        , args(args) {}
};

// Pointer expression: (base_ptr + off).
class ptr_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(ptr_t)

    // off - offset in bytes.
    static expr_t make(const expr_t &base, const expr_t &off) {
        return expr_t(new ptr_t(base, off));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return base.is_equal(other.base) && off.is_equal(other.off);
    }

    size_t get_hash() const override { return ir_utils::get_hash(base, off); }

    // Normalizes (base op off) pointer so that the new base is a variable and
    // off is an offset expression.
    // Example:
    //     Before call: base = (base0 + off0), off = off1
    //     After call:  base = base0, off = off0 + off1
    static void normalize(
            expr_t &base, expr_t &off, op_kind_t op_kind = op_kind_t::_add);

    expr_t base;
    expr_t off;

private:
    ptr_t(const expr_t &base, const expr_t &off)
        : expr_impl_t(base.type()), base(base), off(off) {
        normalize(this->base, this->off);
    }
};

class shuffle_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(shuffle_t)

    static expr_t make(
            const std::vector<expr_t> &vec, const std::vector<int> &idx) {
        if (idx.size() == 1) return vec[idx[0]];
        return expr_t(new shuffle_t(vec, idx));
    }

    static expr_t make(
            const std::vector<expr_t> &_vec, bool find_equal = true) {
        std::vector<expr_t> vec;
        std::vector<int> idx;
        for (auto &v : _vec) {
            bool found = false;
            int size = int(vec.size());
            if (find_equal) {
                for (int i = 0; i < size; i++) {
                    if (v.is_equal(vec[i])) {
                        idx.push_back(i);
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                vec.push_back(v);
                idx.push_back(size);
            }
        }
        return make(vec, idx);
    }

    static expr_t make_broadcast(const expr_t &expr, int elems) {
        ir_assert(expr.type().is_scalar()) << expr;
        return make({expr}, std::vector<int>(elems, 0));
    }

    // Slices the existing shuffle expression. For inputs (S, beg, end) returns
    // (S[beg], S[beg + 1], ..., S[end - 1]) vector.
    static expr_t make(const expr_t &_shuffle, int beg, int end) {
        auto &shuffle = _shuffle.as<shuffle_t>();
        ir_assert(beg >= 0 && beg <= shuffle.elems());
        ir_assert(end >= 0 && end <= shuffle.elems());
        ir_assert(beg < end);
        std::vector<expr_t> vec;
        std::vector<int> idx(end - beg, -1);
        for (int i = beg; i < end; i++) {
            if (idx[i - beg] != -1) continue;
            int old_idx = shuffle.idx[i];
            vec.push_back(shuffle.vec[old_idx]);
            for (int j = i; j < end; j++) {
                if (shuffle.idx[j] == old_idx)
                    idx[j - beg] = int(vec.size()) - 1;
            }
        }
        return make(vec, idx);
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return ir_utils::is_equal(vec, other.vec)
                && ir_utils::is_equal(idx, other.idx);
    }

    size_t get_hash() const override { return ir_utils::get_hash(vec, idx); }

    int elems() const { return int(idx.size()); }

    bool is_vector() const {
        for (int i = 0; i < elems(); i++)
            if (idx[i] != i) return false;
        return true;
    }

    bool is_broadcast() const { return vec.size() == 1; }

    std::vector<expr_t> vec;
    std::vector<int> idx;

private:
    shuffle_t(const std::vector<expr_t> &vec, const std::vector<int> &idx)
        : expr_impl_t(shuffle_type(vec, idx)), vec(vec), idx(idx) {
        ir_assert(idx.size() > 1) << "Unexpected empty or scalar shuffle.";
    }

    static type_t shuffle_type(
            const std::vector<expr_t> &vec, const std::vector<int> &idx) {
        ir_assert(!vec.empty() && !idx.empty());

        auto elem_type = vec[0].type();
        for (auto &v : vec)
            elem_type = common_type(elem_type, v.type());

        for (size_t i = 0; i < idx.size(); i++) {
            ir_assert(idx[i] >= 0 && idx[i] < int(vec.size()))
                    << "Incorrect index.";
            MAYBE_UNUSED(i);
        }

        int elems = int(idx.size());
        return elem_type.with_elems(elems);
    }
};

// Ternary operation: op(a, b, c).
class ternary_op_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(ternary_op_t)

    static expr_t make(op_kind_t op_kind, const expr_t &a, const expr_t &b,
            const expr_t &c) {
        return expr_t(new ternary_op_t(op_kind, a, b, c));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (op_kind == other.op_kind) && a.is_equal(other.a)
                && b.is_equal(other.b) && c.is_equal(other.c);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(op_kind, a, b, c);
    }

    op_kind_t op_kind;
    expr_t a;
    expr_t b;
    expr_t c;

private:
    ternary_op_t(op_kind_t op_kind, const expr_t &a, const expr_t &b,
            const expr_t &c)
        : expr_impl_t(ternary_op_type(op_kind, a, b, c))
        , op_kind(op_kind)
        , a(a)
        , b(b)
        , c(c) {}
};

inline expr_t ternary_mad(const expr_t &a, const expr_t &b, const expr_t &c) {
    return ternary_op_t::make(op_kind_t::_mad, a, b, c);
}

inline expr_t ternary_add3(const expr_t &a, const expr_t &b, const expr_t &c) {
    return ternary_op_t::make(op_kind_t::_add3, a, b, c);
}

// Unary operation: (op a).
class unary_op_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(unary_op_t)

    static expr_t make(op_kind_t op_kind, const expr_t &a) {
        return expr_t(new unary_op_t(op_kind, a));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (op_kind == other.op_kind) && a.is_equal(other.a);
    }

    size_t get_hash() const override { return ir_utils::get_hash(op_kind, a); }

    op_kind_t op_kind;
    expr_t a;

private:
    unary_op_t(op_kind_t op_kind, const expr_t &a)
        : expr_impl_t(unary_op_type(op_kind, a)), op_kind(op_kind), a(a) {}
};

class var_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(var_t)

    static expr_t make(const type_t &type, const std::string &name) {
        return expr_t(new var_t(type, name));
    }

    bool is_equal(const object_impl_t *obj) const override {
        // Do not allow variable cloning.
        return this == obj;
    }

    size_t get_hash() const override { return ir_utils::get_hash(name); }

    std::string name;

private:
    var_t(const type_t &type, const std::string &name)
        : expr_impl_t(type), name(name) {}
};

// Convertor from C++ type to IR expression.
template <typename T>
expr_t to_expr(T value, const type_t &type) {
#define CASE(ir_type, cpp_type) \
    if (type == type_t::ir_type()) return expr_t((cpp_type)value)

    CASE(_bool, bool);
    CASE(f32, float);
    CASE(s16, int16_t);
    CASE(s32, int32_t);
    CASE(s64, int64_t);
    CASE(u16, uint16_t);
    CASE(u32, uint32_t);
    CASE(u64, uint64_t);

#undef CASE

    ir_error_not_expected() << type;

    return expr_t();
}

template <typename T>
expr_t to_expr(T value) {
    return to_expr(value, type_t::from_cpp<T>());
}

inline bool is_binary_op(const expr_t &e) {
    return e.is<binary_op_t>();
}

inline bool is_binary_op(const expr_t &e, op_kind_t op_kind) {
    if (!is_binary_op(e)) return false;
    return e.as<binary_op_t>().op_kind == op_kind;
}

inline bool is_binary_cmp_op(const expr_t &e) {
    if (!is_binary_op(e)) return false;
    return is_cmp_op(e.as<binary_op_t>().op_kind);
}

inline bool is_const(const expr_t &e) {
    return e.is<bool_imm_t>() || e.is<int_imm_t>() || e.is<float_imm_t>();
}

inline bool is_shuffle_const(const expr_t &e) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return false;
    for (auto &v : shuffle->vec)
        if (!is_const(v)) return false;
    return true;
}

inline bool is_var(const expr_t &e) {
    return e.is<var_t>();
}

// Convertor from IR expression to C++ constant.
template <typename T>
T to_cpp(const expr_t &e) {
    ir_assert(is_const(e)) << "Expression must be constant.";

    if (e.is<int_imm_t>()) return (T)e.as<int_imm_t>().value;
    if (e.is<float_imm_t>()) return (T)e.as<float_imm_t>().value;
    if (e.is<bool_imm_t>()) return (T)e.as<bool_imm_t>().value;

    ir_error_not_expected();
    return 0;
}

expr_t operator-(const expr_t &a);

#define DECLARE_BINARY_OPERATOR(op, op_kind) \
    expr_t operator op(const expr_t &a, const expr_t &b);

DECLARE_BINARY_OPERATOR(+, op_kind_t::_add)
DECLARE_BINARY_OPERATOR(-, op_kind_t::_sub)
DECLARE_BINARY_OPERATOR(*, op_kind_t::_mul)
DECLARE_BINARY_OPERATOR(/, op_kind_t::_div)
DECLARE_BINARY_OPERATOR(%, op_kind_t::_mod)
DECLARE_BINARY_OPERATOR(<<, op_kind_t::_shl)
DECLARE_BINARY_OPERATOR(>>, op_kind_t::_shr)

DECLARE_BINARY_OPERATOR(==, op_kind_t::_eq)
DECLARE_BINARY_OPERATOR(!=, op_kind_t::_ne)
DECLARE_BINARY_OPERATOR(>, op_kind_t::_gt)
DECLARE_BINARY_OPERATOR(>=, op_kind_t::_ge)
DECLARE_BINARY_OPERATOR(<, op_kind_t::_lt)
DECLARE_BINARY_OPERATOR(<=, op_kind_t::_le)

DECLARE_BINARY_OPERATOR(&, op_kind_t::_and)

#undef DECLARE_BINARY_OPERATOR

// Returns a shifted pointer with base `a` (pointer) and offset `b` (in bytes).
// shift_ptr(op, a, b) returns &(a op b) in C++ terms (op is either addition or
// subtraction).
expr_t shift_ptr(op_kind_t op_kind, const expr_t &a, const expr_t &b);

// Base class for IR statement objects.
class stmt_impl_t : public object_impl_t {
public:
    IR_DECL_TYPE_ID(stmt_impl_t)
};

// Wrapper for IR statement objects.
class stmt_t : public object_t {
public:
    using object_t::object_t;

    stmt_t() = default;
    stmt_t(const object_t &obj) : object_t(obj) {}
    stmt_t(object_t &&obj) : object_t(obj) {}
    stmt_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    stmt_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    stmt_t append(const stmt_t &s) const;

private:
    void sanity_check() const override {
        ir_assert(dynamic_cast<const stmt_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
};

enum class alloc_kind_t {
    undef,
    grf, // GRF - general register file.
    slm, // SLM - shared local memory.
    global, // Global memory.
};

class alloc_attr_impl_t : public object_impl_t {};

class alloc_attr_t : public object_t {
public:
    using object_t::object_t;

    alloc_attr_t() = default;
    alloc_attr_t(const object_t &obj) : object_t(obj) {}
    alloc_attr_t(object_t &&obj) : object_t(obj) {}
    alloc_attr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    alloc_attr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

private:
    void sanity_check() const override {
        ir_assert(dynamic_cast<const alloc_attr_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
};

// Allocation attribute for GRF.
class grf_alloc_attr_t : public alloc_attr_impl_t {
public:
    IR_DECL_TYPE_ID(grf_alloc_attr_t)

    static alloc_attr_t make(const ngen_proxy::Bundle &bundle) {
        return alloc_attr_t(new grf_alloc_attr_t(bundle));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return bundle == other.bundle;
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(bundle.bundle_id, bundle.bank_id);
    }

    ngen_proxy::Bundle bundle;

private:
    grf_alloc_attr_t(const ngen_proxy::Bundle &bundle) : bundle(bundle) {}
};

// Allocation for SLM and GRF buffers.
// C++ equivalent:
//     {
//         byte *buf = new byte[size];
//         body;
//      }
class alloc_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(alloc_t)

    static stmt_t make(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}, const stmt_t &body = {}) {
        return stmt_t(new alloc_t(buf, size, kind, attr, body));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return buf.is_equal(other.buf) && (size == other.size)
                && (kind == other.kind) && attr.is_equal(other.attr)
                && body.is_equal(other.body);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(buf, size, kind, attr, body);
    }

    expr_t buf;
    int size;
    alloc_kind_t kind;
    alloc_attr_t attr;
    stmt_t body;

private:
    alloc_t(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr, const stmt_t &body)
        : buf(buf), size(size), kind(kind), attr(attr), body(body) {
        ir_assert(buf.type().is_ptr()) << buf;
    }
};

// Store to a GRF buffer.
// C++ equivalent (when value is scalar):
//     *(value_type *)(&buf[off]) = value;
// C++ equivalent (when value is vector):
//     int _stride = (has_default_stride() ? sizeof(scalar_type) : stride);
//     for (int i = 0; i < elems; i++) {
//         *(scalar_type *)(&buf[off + i * _stride]) = value[i];
//     }
class store_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(store_t)

    // offset and stride are expressed in bytes.
    // default stride means unit stride (in terms of value.type().scalar()
    // elements).
    static stmt_t make(const expr_t &buf, const expr_t &off,
            const expr_t &value, int stride = default_stride,
            const expr_t &mask = expr_t()) {
        return stmt_t(new store_t(buf, off, value, stride, mask));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return buf.is_equal(other.buf) && off.is_equal(other.off)
                && value.is_equal(other.value) && (stride == other.stride)
                && mask.is_equal(other.mask);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(buf, off, value, stride, mask);
    }

    bool has_default_stride() const { return stride == default_stride; }

    static const int default_stride = -1;

    expr_t buf;
    expr_t off;
    expr_t value;
    int stride;
    expr_t mask;

private:
    store_t(const expr_t &_buf, const expr_t &_off, const expr_t &_value,
            int _stride, const expr_t &_mask)
        : buf(_buf), off(_off), value(_value), stride(_stride), mask(_mask) {
        normalize_ptr(value.type(), buf, off);
        ir_assert(is_var(buf)) << buf;
        ir_assert(buf.type().is_ptr()) << buf;
        if (stride == value.type().scalar().size()) stride = default_stride;
        if (!mask.is_empty())
            ir_assert(mask.type() == type_t::_bool(value.type().elems()));
    }
};

// Loop statement with unit increment.
// C++ equivalent:
//    for (var = init; var < bound; var++) {
//        body;
//    }
// unroll specifies the unroll factor, unroll = 1 means no unrolling.
class for_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(for_t)

    static stmt_t make(const expr_t &var, const expr_t &init,
            const expr_t &bound, const stmt_t &body = {}, int unroll = 1) {
        return stmt_t(new for_t(var, init, bound, body, unroll));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return var.is_equal(other.var) && init.is_equal(other.init)
                && bound.is_equal(other.bound) && body.is_equal(other.body)
                && (unroll == other.unroll);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(var, init, bound, body, unroll);
    }

    expr_t var;
    expr_t init;
    expr_t bound;
    stmt_t body;
    int unroll;

private:
    for_t(const expr_t &var, const expr_t &init, const expr_t &bound,
            const stmt_t &body, int unroll)
        : var(var), init(init), bound(bound), body(body), unroll(unroll) {}
};

// If-else statement.
// C++ equivalent:
//     if (cond) {
//         body;
//     } else {
//         else_body;
//     }
class if_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(if_t)

    static stmt_t make(const expr_t &cond, const stmt_t &body,
            const stmt_t &else_body = stmt_t()) {
        return stmt_t(new if_t(cond, body, else_body));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return cond.is_equal(other.cond) && body.is_equal(other.body)
                && else_body.is_equal(other.else_body);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(cond, body, else_body);
    }

    expr_t cond;
    stmt_t body;
    stmt_t else_body;

private:
    if_t(const expr_t &cond, const stmt_t &body, const stmt_t &else_body)
        : cond(cond), body(body), else_body(else_body) {}
};

// Let statement, used to bind a variable to a value within a scope.
// C++ equivalent:
//     {
//         var = value;
//         body;
//     }
class let_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(let_t)

    static stmt_t make(
            const expr_t &var, const expr_t &value, const stmt_t &body = {}) {
        return stmt_t(new let_t(var, value, body));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return var.is_equal(other.var) && value.is_equal(other.value)
                && body.is_equal(other.body);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(var, value, body);
    }

    expr_t var;
    expr_t value;
    stmt_t body;

private:
    let_t(const expr_t &var, const expr_t &value, const stmt_t &body)
        : var(var), value(value), body(body) {}
};

// Statement label, specific to GEMM/convolution.
class stmt_label_t {
public:
    static stmt_label_t kernel(int index = -1) {
        return stmt_label_t(kind_t::_kernel, index);
    }
    static stmt_label_t compute_loop(int index = -1) {
        return stmt_label_t(kind_t::_compute_loop, index);
    }
    static stmt_label_t c_store(int index = -1) {
        return stmt_label_t(kind_t::_c_store, index);
    }
    static stmt_label_t c_zero_out(int index = -1) {
        return stmt_label_t(kind_t::_c_zero_out, index);
    }
    static stmt_label_t b_reduced_zero_out(int index = -1) {
        return stmt_label_t(kind_t::_b_reduced_zero_out, index);
    }
    static stmt_label_t g2s_load(int index = -1) {
        return stmt_label_t(kind_t::_g2s_load, index);
    }
    static stmt_label_t g2s_store(int index = -1) {
        return stmt_label_t(kind_t::_g2s_store, index);
    }
    static stmt_label_t g2r_load(int index = -1) {
        return stmt_label_t(kind_t::_g2r_load, index);
    }
    static stmt_label_t s2r_load(int index = -1) {
        return stmt_label_t(kind_t::_s2r_load, index);
    }
    static stmt_label_t prefetch(int index = -1) {
        return stmt_label_t(kind_t::_prefetch, index);
    }
    static stmt_label_t mul(int index = -1) {
        return stmt_label_t(kind_t::_mul, index);
    }

    bool operator==(const stmt_label_t &other) const {
        if (kind_ != other.kind_) return false;
        if (index_ == -1 || other.index_ == -1) return true;
        return index_ == other.index_;
    }

    size_t get_hash() const { return ir_utils::get_hash(kind_, index_); }

    std::string str() const {
        switch (kind_) {
#define CASE(kind) \
    case kind_t::_##kind: return #kind
            CASE(kernel);
            CASE(compute_loop);
            CASE(c_store);
            CASE(c_zero_out);
            CASE(g2r_load);
            CASE(g2s_load);
            CASE(g2s_store);
            CASE(s2r_load);
            CASE(prefetch);
            CASE(mul);
#undef CASE
            default: ir_error_not_expected();
        }
        return {};
    }

private:
    enum class kind_t {
        _undef,
        _kernel, // All kernel.
        _compute_loop, // Compute loop.
        _c_store, // GRF to GMEM store of C.
        _c_zero_out, // Zeroing-out of C.
        _b_reduced_zero_out, // Zeroing-out of B reduced buffer.
        _g2r_load, // GMEM to GRF load for further multiplication.
        _g2s_load, // GMEM to GRF load for GMEM -> SLM copy.
        _g2s_store, // GRF to SLM store for GMEM -> SLM copy.
        _s2r_load, // SLM to GRF load for further multiplication.
        _prefetch, // GMEM prefetch.
        _mul, // Multiplication.
    };

    stmt_label_t() : kind_(kind_t::_undef), index_(-1) {}
    stmt_label_t(kind_t kind, int index) : kind_(kind), index_(index) {}

    kind_t kind_;
    int index_; // Used to differentiate groups with the same kind.
};

inline std::ostream &operator<<(std::ostream &out, const stmt_label_t &label) {
    out << label.str();
    return out;
}

// Statement group, used to assign a label to a group of statements.
class stmt_group_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(stmt_group_t)

    static stmt_t make(const stmt_label_t &label, const stmt_t &body) {
        return stmt_t(new stmt_group_t(label, body));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (label == other.label) && body.is_equal(other.body);
    }

    size_t get_hash() const override { return ir_utils::get_hash(label, body); }

    stmt_label_t label;
    stmt_t body;

private:
    stmt_group_t(const stmt_label_t &label, const stmt_t &body)
        : label(label), body(body) {}
};

// Statement sequence, allows combining two statements.
// C++ equivalent:
//     {
//         head;
//         tail;
//     }
class stmt_seq_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(stmt_seq_t)

    static stmt_t make(const stmt_t &head, const stmt_t &tail) {
        return stmt_t(new stmt_seq_t(head, tail));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return head.is_equal(other.head) && tail.is_equal(other.tail);
    }

    size_t get_hash() const override { return ir_utils::get_hash(head, tail); }

    stmt_t head;
    stmt_t tail;

private:
    stmt_seq_t(const stmt_t &head, const stmt_t &tail)
        : head(head), tail(tail) {}
};

inline stmt_t stmt_t::append(const stmt_t &s) const {
    if (is_empty()) return s;
    return stmt_seq_t::make(*this, s);
}

// Function call attribute.
class func_call_attr_impl_t : public object_impl_t {};

class func_call_attr_t : public object_t {
public:
    using object_t::object_t;

    func_call_attr_t() = default;
    func_call_attr_t(const object_t &obj) : object_t(obj) {}
    func_call_attr_t(object_t &&obj) : object_t(obj) {}
    func_call_attr_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    func_call_attr_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    // Returns a function call with the attribute applied. The input statement
    // must be a function call.
    stmt_t apply_to(const stmt_t &s) const;

private:
    void sanity_check() const override {
        ir_assert(dynamic_cast<const func_call_attr_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
};

// Instruction modifier, relies on nGEN API.
class instruction_modifier_attr_t : public func_call_attr_impl_t {
public:
    IR_DECL_TYPE_ID(instruction_modifier_attr_t)

    static func_call_attr_t make(const ngen_proxy::InstructionModifier &mod) {
        return func_call_attr_t(new instruction_modifier_attr_t(mod));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return mod == other.mod;
    }

    size_t get_hash() const override { return ir_utils::get_hash(mod); }

    std::string str() const override {
        std::ostringstream oss;
        oss << "{";
        bool is_first = true;
        auto append = [&](const std::string &s) {
            if (!is_first) oss << ", ";
            oss << s;
            is_first = false;
        };
        if (mod.is_atomic) append("Atomic");
        if (!mod.sbid.is_empty()) {
            append(std::string("$") + std::to_string(mod.sbid.token));
        }
        oss << "}";
        return oss.str();
    }

    ngen_proxy::InstructionModifier mod;

private:
    instruction_modifier_attr_t(const ngen_proxy::InstructionModifier &mod)
        : mod(mod) {}
};

// Base class for function IR objects.
class func_impl_t : public object_impl_t {
public:
    IR_DECL_TYPE_ID(func_impl_t)

    stmt_t call(const std::vector<expr_t> &args,
            const func_call_attr_t &attr = {}) const;
};

// Wrapper for IR function objects.
class func_t : public object_t {
public:
    using object_t::object_t;

    func_t() = default;
    func_t(const object_t &obj) : object_t(obj) {}
    func_t(object_t &&obj) : object_t(obj) {}
    func_t &operator=(const object_t &obj) {
        object_t::operator=(obj);
        return *this;
    }
    func_t &operator=(object_t &&obj) {
        object_t::operator=(obj);
        return *this;
    }

    stmt_t call(const std::vector<expr_t> &args = {},
            const func_call_attr_t &attr = {}) const {
        return ((const func_impl_t *)impl())->call(args, attr);
    }

private:
    void sanity_check() const override {
        ir_assert(dynamic_cast<const func_impl_t *>(impl()) == impl())
                << object_t(impl());
    }
};

// Function call.
class func_call_t : public stmt_impl_t {
public:
    IR_DECL_STMT_TYPE_ID(func_call_t)

    static stmt_t make(const func_t &func, const std::vector<expr_t> &args,
            const func_call_attr_t &attr = {}) {
        return stmt_t(new func_call_t(func, args, attr));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return func.is_equal(other.func) && ir_utils::is_equal(args, other.args)
                && attr.is_equal(other.attr);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(func, args, attr);
    }

    func_t func;
    std::vector<expr_t> args;
    func_call_attr_t attr;

private:
    func_call_t(const func_t &func, const std::vector<expr_t> &args,
            const func_call_attr_t &attr)
        : func(func), args(args), attr(attr) {
        ir_assert(!func.is_empty());
    }
};

inline stmt_t func_impl_t::call(
        const std::vector<expr_t> &args, const func_call_attr_t &attr) const {
    return func_call_t::make(this, args, attr);
}

inline stmt_t func_call_attr_t::apply_to(const stmt_t &s) const {
    auto &c = s.as<func_call_t>();
    ir_assert(c.attr.is_empty())
            << "Merging of attributes is not supported: " << s;
    return func_call_t::make(c.func, c.args, *this);
}

template <typename F>
inline bool is_func_call(const stmt_t &s) {
    auto *c = s.as_ptr<func_call_t>();
    if (!c) return false;
    return c->func.is<F>();
}

// Generic function with a name.
class builtin_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(builtin_t, func_impl_t)

    static func_t make(const std::string &name) {
        return func_t(new builtin_t(name));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return name == other.name;
    }

    size_t get_hash() const override { return ir_utils::get_hash(name); }

    std::string str() const override { return name; }

    std::string name;

private:
    builtin_t(const std::string &name) : name(name) {}
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
