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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/math_utils.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ir_utils;

// Generic pattern expression, used as a wild card during pattern matching. Can
// match any expression.
class pexpr_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(pexpr_t)

    static expr_t make(int id) { return expr_t(new pexpr_t(id)); }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return id == other.id;
    }

    size_t get_hash() const override { return ir_utils::get_hash(id); }

    std::string str() const override {
        std::ostringstream oss;
        oss << "pexpr_t(" << id << ")";
        return oss.str();
    }

    static expr_t x() { return pexpr_t::make(0); }
    static expr_t y() { return pexpr_t::make(1); }
    static expr_t z() { return pexpr_t::make(2); }

    int id;

private:
    pexpr_t(int id) : expr_impl_t(type_t::undef()), id(id) {}
};

// Pattern expression for int_imm_t, used as a wild card during pattern
// matching. Can match any int_imm_t with the given value.
class pint_imm_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(pint_imm_t)

    // Matches an integer constant with the given value.
    static expr_t make(int64_t value) {
        return expr_t(new pint_imm_t(-1, value));
    }

    // Matches any integer constant.
    static expr_t make_any(int64_t id) { return expr_t(new pint_imm_t(id, 0)); }

    bool matches(const int_imm_t &imm) const {
        if (id == -1) return value == imm.value;
        return true;
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return value == other.value;
    }

    size_t get_hash() const override { return ir_utils::get_hash(value); }

    int id;
    int64_t value;

private:
    pint_imm_t(int id, int64_t value)
        : expr_impl_t(type_t::undef()), id(id), value(value) {}
};

// Stores already matched pairs of <pattern expression, matched expression>.
class match_context_t {
public:
    bool contains(const expr_t &ptrn) const {
        return expr_matched_.count(ptrn) != 0;
    }

    void set(const expr_t &ptrn, const expr_t &e) {
        ir_assert(ptrn.is<pexpr_t>());
        auto ret = expr_matched_.insert({ptrn, e});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
    }

    const expr_t &operator[](const expr_t &ptrn) const {
        return expr_matched_.at(ptrn);
    }

    template <typename T>
    const T &at(const expr_t &ptrn) const {
        return expr_matched_.at(ptrn).as<T>();
    }

    expr_t sub(const expr_t &expr) const;

private:
    object_eq_map_t<expr_t, expr_t> expr_matched_;
};

class pexpr_substitute_t : public ir_mutator_t {
public:
    using ir_mutator_t::_mutate;

    pexpr_substitute_t(const match_context_t *ctx) : ctx_(ctx) {}

    dispatch_func_type find_dispatch_func(int64_t ti) const override {
        if (ti == pexpr_t::_dispatch_type_id())
            return &pexpr_substitute_t::call<pexpr_t>;
        return ir_mutator_t::find_dispatch_func(ti);
    }

    object_t _mutate(const pexpr_t *obj) { return (*ctx_)[expr_t(obj)]; }

private:
    template <typename T>
    static object_t call(ir_mutator_t *mutator, const object_impl_t *obj) {
        auto *this_mutator = (pexpr_substitute_t *)mutator;
        return this_mutator->_mutate((const T *)obj);
    }

    const match_context_t *ctx_;
};

// Replaces occurrences of pattern expressions in `expr` according to the
// context.
expr_t match_context_t::sub(const expr_t &expr) const {
    pexpr_substitute_t s(this);
    return s.mutate(expr);
}

// Returns true if the expression matches the pattern, false otherwise. Upon
// successful match the context contains information matched pattern
// expressions.
bool match(const expr_t &ptrn, const expr_t &expr, match_context_t &ctx);

bool match_binary(
        const expr_t &ptrn, const expr_t &expr, match_context_t &ctx) {
    bool ptrn_is_binary = is_binary_op(ptrn);
    bool expr_is_binary = is_binary_op(expr);

    if (!ptrn_is_binary || !expr_is_binary) return false;

    auto &ptrn_op = ptrn.as<binary_op_t>();
    auto &expr_op = expr.as<binary_op_t>();
    if (ptrn_op.op_kind != expr_op.op_kind) return false;

    match_context_t ctx_copy = ctx;
    if (match(ptrn_op.a, expr_op.a, ctx_copy)
            && match(ptrn_op.b, expr_op.b, ctx_copy)) {
        ctx = ctx_copy;
        return true;
    }
    return false;
}

bool match_iif(const expr_t &ptrn, const expr_t &expr, match_context_t &ctx) {
    bool ptrn_is_iif = ptrn.is<iif_t>();
    bool expr_is_iif = expr.is<iif_t>();

    if (!ptrn_is_iif || !expr_is_iif) return false;

    auto &ptrn_iif = ptrn.as<iif_t>();
    auto &expr_iif = expr.as<iif_t>();

    match_context_t ctx_copy = ctx;
    if (match(ptrn_iif.cond, expr_iif.cond, ctx_copy)
            && match(ptrn_iif.true_expr, expr_iif.true_expr, ctx_copy)
            && match(ptrn_iif.false_expr, expr_iif.false_expr, ctx_copy)) {
        ctx = ctx_copy;
        return true;
    }

    return false;
}

bool match(const expr_t &ptrn, const expr_t &expr, match_context_t &ctx) {
    if (ptrn.is_equal(expr)) return true;

    if (ptrn.is<pint_imm_t>()) {
        if (!expr.is<int_imm_t>()) return false;

        auto &expr_imm = expr.as<int_imm_t>();
        auto &ptrn_imm = ptrn.as<pint_imm_t>();
        return ptrn_imm.matches(expr_imm);
    }

    if (ptrn.is<pexpr_t>()) {
        if (ctx.contains(ptrn)) {
            if (!ctx[ptrn].is_equal(expr)) return false;
        } else {
            ctx.set(ptrn, expr);
        }
        return true;
    }

    if (match_binary(ptrn, expr, ctx)) return true;
    if (match_iif(ptrn, expr, ctx)) return true;

    return false;
}

// Rewrites expression `expr` according to `from` -> `to` rule.
// Example:
//     auto x = pexpr_t::x();
//     auto c = rewrite(a + a, x + x, 2 * x);
//     // Now c is equal to (2 * a).
expr_t rewrite(const expr_t &expr, const expr_t &from, const expr_t &to,
        bool *rewritten = nullptr) {
    match_context_t ctx;
    if (match(from, expr, ctx)) {
        if (rewritten) *rewritten = true;
        return ctx.sub(to);
    }
    if (rewritten) *rewritten = false;
    return expr;
}

expr_t simplify_try_rules(const expr_t &_e) {
    static auto x = pexpr_t::x();
    static auto y = pexpr_t::y();
    static auto z = pexpr_t::z();

    static auto _0 = pint_imm_t::make(0);
    static auto _1 = pint_imm_t::make(1);
    static auto _2 = pint_imm_t::make(2);

    auto e = _e;

#define REWRITE(a, b) \
    do { \
        bool rewritten; \
        static auto _a = a; \
        static auto _b = b; \
        e = rewrite(e, _a, _b, &rewritten); \
        if (rewritten) return e; \
    } while (false)

#define REWRITE_NO_STATIC(a, b) \
    do { \
        bool rewritten; \
        e = rewrite(e, a, b, &rewritten); \
        if (rewritten) return e; \
    } while (false)

    // Addition rules.
    REWRITE(x + _0, x);
    REWRITE(_0 + x, x);
    REWRITE(x + x, 2 * x);

    // Subtraction rules.
    REWRITE(x - _0, x);
    REWRITE(_0 - x, -x);
    REWRITE(x - x, 0);

    // Multiplication rules.
    REWRITE(x * _0, 0);
    REWRITE(_0 * x, 0);
    REWRITE(x * _1, x);
    REWRITE(_1 * x, x);

    // Division rules.
    REWRITE(_0 / x, 0);
    REWRITE(x / _1, x);
    REWRITE(x / x, 1);
    REWRITE(x * y / y, x);

    REWRITE(x * y % y, 0);
    REWRITE(y * x % y, 0);

    REWRITE(x % _1, 0);
    REWRITE(0 % x, 0);

    // Ternary operation rules.
    REWRITE(iif_t::make(expr_t(true), x, y), x);
    REWRITE(iif_t::make(expr_t(false), x, y), y);
    REWRITE(iif_t::make(x, y, y), y);

    // Boolean rules.
    if (e.type().is_bool()) {
        auto _true = (e.type().is_scalar()
                        ? expr_t(true)
                        : shuffle_t::make_broadcast(
                                expr_t(true), e.type().elems()));
        auto _false = (e.type().is_scalar()
                        ? expr_t(false)
                        : shuffle_t::make_broadcast(
                                expr_t(false), e.type().elems()));
        REWRITE_NO_STATIC(_true & x, x);
        REWRITE_NO_STATIC(x & _true, x);
        REWRITE_NO_STATIC(_false & x, _false);
        REWRITE_NO_STATIC(x & _false, _false);
    }

    return e;
}

expr_t simplify_try_ternary_rules(const expr_t &_e) {
    static auto x = pexpr_t::x();
    static auto y = pexpr_t::y();
    static auto z = pexpr_t::z();

    auto e = _e;

    // add3 rules.
    REWRITE((x + y) + z, ternary_add3(x, y, z));
    REWRITE(x + (y + z), ternary_add3(x, y, z));

    // mad rules.
    REWRITE(x + y * z, ternary_mad(x, y, z));
    REWRITE(x - y * z, ternary_mad(x, -y, z));
    REWRITE(y * z + x, ternary_mad(x, y, z));
    REWRITE(y * z - x, ternary_mad(-x, y, z));

    return e;
}

#undef REWRITE
#undef REWRITE_NO_STATIC

class term_rewrite_transformer_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t *obj) override {
        return mutate_expr(obj);
    }
    object_t _mutate(const iif_t *obj) override { return mutate_expr(obj); }

    template <typename T>
    expr_t mutate_expr(const T *obj) {
        auto e_old = ir_mutator_t::_mutate(obj);
        auto e = simplify_try_rules(e_old);
        if (e.is_same(e_old)) return e_old;
        return mutate(e);
    }
};

// Simplifies expression using rewriting rules.
expr_t simplify_rewrite(const expr_t &e) {
    expr_t ret;
    if (is_const(e) || is_var(e)) {
        ret = e;
    } else {
        term_rewrite_transformer_t trt;
        ret = trt.mutate(e);
    }
    return ret;
}

class ternary_rewrite_transformer_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t *obj) override {
        return mutate_expr(obj);
    }
    object_t _mutate(const iif_t *obj) override { return mutate_expr(obj); }

    template <typename T>
    expr_t mutate_expr(const T *obj) {
        auto e_old = ir_mutator_t::_mutate(obj);
        auto e = simplify_try_ternary_rules(e_old);
        if (e.is_same(e_old)) return e_old;
        return mutate(e);
    }
};

expr_t simplify_rewrite_with_ternary(const expr_t &e, bool recursive) {
    expr_t ret;
    if (is_const(e) || is_var(e)) {
        ret = e;
    } else if (!recursive) {
        ret = simplify_try_ternary_rules(e);
    } else {
        ternary_rewrite_transformer_t trt;
        ret = trt.mutate(e);
    }
    return ret;
}

class cmp_simplifier_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t *obj) override {
        auto e = ir_mutator_t::_mutate(obj);
        if (!is_binary_cmp_op(e)) return e;

        e = simplify_mod_comparison(e);

        return e;
    }

    static expr_t reduce_lhs_rhs(const expr_t &e) {
        if (!is_binary_cmp_op(e)) return e;

        auto &op = e.as<binary_op_t>();

        // Rule:
        //     (c0 * x op c1) or (x * c0 op c1) ->
        //     x new_op (c1 / c0) if abs(c1) % abs(c0) == 0
        //     new_op == op or new_op == negate_cmp_op(op)
        expr_t c0;
        expr_t c1 = op.b;
        expr_t x;

        if (!is_const(c1)) return e;
        if (!is_binary_op(op.a, op_kind_t::_mul)) return e;

        auto &a_op = op.a.as<binary_op_t>();
        if (is_const(a_op.a)) {
            c0 = a_op.a;
            x = a_op.b;
        } else if (is_const(a_op.b)) {
            x = a_op.a;
            c0 = a_op.b;
        }

        if (c0.is_empty()) return e;
        if (!c0.type().is_int()) return e;
        if (!c1.type().is_int()) return e;

        auto i_c0 = to_cpp<int64_t>(c0);
        auto i_c1 = to_cpp<int64_t>(c1);

        bool is_c0_neg = (i_c0 < 0);
        bool sign = ((i_c0 < 0) != (i_c1 < 0));
        i_c0 = std::abs(i_c0);
        i_c1 = std::abs(i_c1);

        bool has_mod = (i_c1 % i_c0 != 0);
        if (has_mod
                && utils::one_of(op.op_kind, op_kind_t::_eq, op_kind_t::_ne))
            return e;

        auto new_op_kind = (is_c0_neg ? negate_cmp_op(op.op_kind) : op.op_kind);
        int64_t div = i_c1 / i_c0;
        if (has_mod) {
            switch (new_op_kind) {
                case op_kind_t::_ge:
                case op_kind_t::_gt:
                    new_op_kind = op_kind_t::_ge;
                    div = (sign ? div : div + 1);
                    break;
                case op_kind_t::_le:
                case op_kind_t::_lt:
                    new_op_kind = op_kind_t::_le;
                    div = (sign ? div + 1 : div);
                    break;
                default: ir_error_not_expected();
            }
        }

        return binary_op_t::make(new_op_kind, x, (sign ? -1 : 1) * div);
    }

    static expr_t simplify_mod_comparison(const expr_t &e) {
        if (!is_binary_cmp_op(e)) return e;

        auto &op = e.as<binary_op_t>();

        // Use the following inequalities:
        //     0 <= (x % c0) < c0
        if (!is_binary_op(op.a, op_kind_t::_mod)) return e;
        if (!is_const(op.b)) return e;

        auto &a_op = op.a.as<binary_op_t>();
        if (!is_const(a_op.b)) return e;

        auto &c0 = a_op.b;
        ir_assert(to_cpp<int64_t>(c0) > 0) << e;

        // Comparison against a constant is a continuous function, just check
        // boundary points.
        auto cond0 = binary_op_t::make(op.op_kind, 0, op.b);
        auto cond1 = binary_op_t::make(op.op_kind, c0 - 1, op.b);

        bool is_cond0 = to_cpp<bool>(const_fold_non_recursive(cond0));
        bool is_cond1 = to_cpp<bool>(const_fold_non_recursive(cond1));

        // Conditions are equal, can prove.
        if (is_cond0 == is_cond1) return expr_t(is_cond0);

        // Can't prove, return the original expression.
        return e;
    }
};

expr_t simplify_comparison(const expr_t &e) {
    return cmp_simplifier_t().mutate(e);
}

class range_simplifier_t : public ir_mutator_t {
public:
    range_simplifier_t(const constraint_set_t &cset) : cset(cset) {}

    object_t _mutate(const var_t *obj) override {
        expr_t value;
        if (cset.is_single_value(obj, value)) return std::move(value);
        return obj;
    }

    const constraint_set_t &cset;
};

// Finds all constant operands on an N-ary operation, returns the folded
// constant in `const_arg` and the remaining operands in `other_args`.
void split_const_nary_op_arg(op_kind_t op_kind, const std::vector<expr_t> &args,
        expr_t &const_arg, std::vector<expr_t> &other_args) {
    other_args.resize(0);

    const_arg = expr_t();
    for (auto &a : args) {
        if (is_const(a)) {
            if (const_arg.is_empty()) {
                const_arg = a;
                continue;
            }
            const_arg = const_fold_non_recursive(
                    binary_op_t::make(op_kind, const_arg, a));
        } else {
            other_args.push_back(a);
        }
    }
}

// Folds all constant operands into one.
void fold_const_nary_op_args(op_kind_t op_kind, const std::vector<expr_t> &args,
        std::vector<expr_t> &new_args) {
    expr_t c;
    split_const_nary_op_arg(op_kind, args, c, new_args);
    if (c.is_empty()) return;
    if (op_kind == op_kind_t::_mul && is_zero(c)) {
        new_args.clear();
        new_args.push_back(c);
        return;
    }
    if (op_kind == op_kind_t::_mul && is_one(c)) return;
    if (op_kind == op_kind_t::_add && is_zero(c)) return;
    new_args.push_back(c);
}

expr_t cvt_mul_to_nary_op(const expr_t &a, const expr_t &b) {
    auto *a_nary = a.as_ptr<nary_op_t>();
    auto *b_nary = b.as_ptr<nary_op_t>();

    if (a_nary) ir_assert(a_nary->op_kind == op_kind_t::_mul);
    if (b_nary) ir_assert(b_nary->op_kind == op_kind_t::_mul);

    auto a_args = cvt_expr_to_nary_op_args(a);
    auto b_args = cvt_expr_to_nary_op_args(b);

    std::vector<expr_t> args;
    args.insert(args.end(), a_args.begin(), a_args.end());
    args.insert(args.end(), b_args.begin(), b_args.end());
    return make_nary_op(op_kind_t::_mul, args);
}

class nary_op_visitor_t : public ir_visitor_t {
public:
    using ir_visitor_t::_visit;

    virtual dispatch_func_type find_dispatch_func(int64_t ti) const {
        if (ti == nary_op_t::_dispatch_type_id())
            return &nary_op_visitor_t::call<nary_op_t>;
        return ir_visitor_t::find_dispatch_func(ti);
    }

    virtual void _visit(const nary_op_t *obj) { visit(obj->args); }

private:
    template <typename T>
    static void call(ir_visitor_t *visitor, const object_impl_t *obj) {
        auto *this_visitor = (nary_op_visitor_t *)visitor;
        this_visitor->_visit((const nary_op_t *)obj);
    }
};

class nary_op_mutator_t : public ir_mutator_t {
public:
    using ir_mutator_t::_mutate;

    virtual object_t _mutate(const nary_op_t *obj) {
        auto args = mutate(obj->args);
        if (ir_utils::is_equal(args, obj->args)) return obj;
        return make_nary_op(obj->op_kind, args);
    }

protected:
    dispatch_func_type find_dispatch_func(int64_t ti) const override {
        if (ti == nary_op_t::_dispatch_type_id())
            return &nary_op_mutator_t::call<nary_op_t>;
        return ir_mutator_t::find_dispatch_func(ti);
    }

private:
    template <typename T>
    static object_t call(ir_mutator_t *mutator, const object_impl_t *obj) {
        auto *this_mutator = (nary_op_mutator_t *)mutator;
        return this_mutator->_mutate((const nary_op_t *)obj);
    }
};

class nary_op_transformer_t : public nary_op_mutator_t {
public:
    using nary_op_mutator_t::_mutate;

    object_t _mutate(const binary_op_t *obj) override {
        // Skip vector types.
        if (!obj->type.is_scalar()) return nary_op_mutator_t::_mutate(obj);
        switch (obj->op_kind) {
            case op_kind_t::_add:
            case op_kind_t::_sub:
            case op_kind_t::_mul: {
                auto a = mutate(obj->a);
                auto b = mutate(obj->b);
                std::vector<expr_t> args = {a, b};
                auto nary_op_kind = obj->op_kind;
                if (obj->op_kind == op_kind_t::_sub) {
                    nary_op_kind = op_kind_t::_add;
                    args[1] *= -1;
                }
                return mutate(make_nary_op(nary_op_kind, args));
            }
            default: return nary_op_mutator_t::_mutate(obj);
        }
    }
};

class nary_op_flattener_t : public nary_op_mutator_t {
public:
    object_t _mutate(const nary_op_t *obj) override {
        std::vector<expr_t> args;
        for (auto &a : obj->args) {
            auto new_a = mutate(a);
            auto *nary = new_a.as_ptr<nary_op_t>();
            if (nary && nary->op_kind == obj->op_kind) {
                args.insert(args.end(), nary->args.begin(), nary->args.end());
            } else {
                args.push_back(new_a);
            }
        }
        return make_nary_op(obj->op_kind, args);
    }
};

expr_t nary_op_flatten(const expr_t &e) {
    return nary_op_flattener_t().mutate(e);
}

class mul_nary_op_expander_t : public nary_op_mutator_t {
public:
    object_t _mutate(const nary_op_t *obj) override {
        if (obj->op_kind != op_kind_t::_mul) {
            return nary_op_flatten(nary_op_mutator_t::_mutate(obj));
        }

        auto args = mutate(obj->args);
        std::vector<expr_t> new_args;
        for (size_t i = 0; i < args.size(); i++) {
            auto *nary = args[i].as_ptr<nary_op_t>();
            if (nary && nary->op_kind != op_kind_t::_add) {
                ir_error_not_expected();
            }
            auto i_args = cvt_expr_to_nary_op_args(args[i]);
            if (new_args.empty()) {
                new_args = i_args;
                continue;
            }
            std::vector<expr_t> next_args;
            for (auto &a : new_args)
                for (auto &b : i_args)
                    next_args.push_back(cvt_mul_to_nary_op(a, b));

            new_args = next_args;
        }
        return nary_op_flatten(make_nary_op(op_kind_t::_add, new_args));
    }
};

class nary_op_canonical_verifier_t : public nary_op_visitor_t {
public:
    bool is_canonical() const { return is_canonical_; }

    void _visit(const binary_op_t *obj) override {
        // Skip vector types.
        if (!obj->type.is_scalar()) {
            visit_new_scope(obj);
            return;
        }
        switch (obj->op_kind) {
            // These operations must be converted to nary_op_t at this point.
            case op_kind_t::_add:
            case op_kind_t::_sub:
            case op_kind_t::_mul: set_canonical_false(); break;
            default: {
                // Assume new scope here, n-ary operations from different
                // scopes can't be merged.
                visit_new_scope(obj);
                break;
            }
        }
    }

    void _visit(const iif_t *obj) override { visit_new_scope(obj); }

    void _visit(const load_t *obj) override { visit_new_scope(obj); }

    void _visit(const ptr_t *obj) override { visit_new_scope(obj); }

    void _visit(const nary_op_t *obj) override {
        if (parent_nary_) {
            if (!(parent_nary_->op_kind == op_kind_t::_add
                        && obj->op_kind == op_kind_t::_mul)) {
                // Multiplications must be expanded at this point.
                set_canonical_false();
                return;
            }
        }

        auto *old_parent_nary = parent_nary_;
        parent_nary_ = obj;
        visit(obj->args);
        parent_nary_ = old_parent_nary;
    }

private:
    void set_canonical_false() { is_canonical_ = false; }

    template <typename T>
    void visit_new_scope(const T *obj) {
        auto *old_parent_nary = parent_nary_;
        parent_nary_ = nullptr;
        nary_op_visitor_t::_visit(obj);
        parent_nary_ = old_parent_nary;
    }

    bool is_canonical_ = true;
    const nary_op_t *parent_nary_ = nullptr;
};

// Checks if the expression is in the canonical N-ary form.
bool is_nary_op_canonical(const expr_t &e) {
    nary_op_canonical_verifier_t v;
    v.visit(e);
    return v.is_canonical();
}

class nary_op_back_transformer_t : public nary_op_mutator_t {
public:
    object_t _mutate(const nary_op_t *obj) {
        auto new_obj = nary_op_mutator_t::_mutate(obj);
        auto &nary = new_obj.as<nary_op_t>();
        ir_assert(nary.args.size() > 0) << new_obj;

        if (nary.args.size() == 1) return nary.args[0];

        if (nary.op_kind == op_kind_t::_add) {
            expr_t ret = nary.args[0] + nary.args[1];
            for (size_t i = 2; i < nary.args.size(); i++)
                ret += nary.args[i];
            return std::move(ret);
        } else if (nary.op_kind == op_kind_t::_mul) {
            expr_t ret = nary.args[0] * nary.args[1];
            for (size_t i = 2; i < nary.args.size(); i++)
                ret *= nary.args[i];
            return std::move(ret);
        }
        ir_error_not_expected();
        return expr_t();
    }
};

// Stores factorization of an expression in the canonical (normalized) form:
//     expr = (f(0), f(1), f(2), ... f(n))
// f(0), ... f(n-1) are non-constant expressions, f(n) is a constant.
class factored_expr_t : public expr_impl_t {
public:
    IR_DECL_EXPR_TYPE_ID(factored_expr_t);

    static expr_t make(const expr_t &e) {
        return expr_t(new factored_expr_t(e));
    }

    static expr_t make(const type_t &type, const std::vector<expr_t> &factors) {
        return expr_t(new factored_expr_t(type, factors));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        if (factors.size() != other.factors.size()) return false;
        if (!factors.back().is_equal(other.factors.back())) return false;

        auto common = intersect(obj);
        auto &f_common = common.as<factored_expr_t>();
        return f_common.factors.size() == factors.size();
    }

    // Constant factor is ignored during comparison.
    bool is_equal_ignore_const(const object_impl_t *obj) const {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        if (factors.size() != other.factors.size()) return false;

        auto common = intersect_ignore_const(obj);
        auto &f_common = common.as<factored_expr_t>();
        return f_common.factors.size() == factors.size();
    }

    size_t get_hash() const override { return ir_utils::get_hash(factors); }

    std::string str() const override {
        std::ostringstream oss;
        oss << "f(";
        for (size_t i = 0; i < factors.size(); i++) {
            oss << (i != 0 ? " x " : "") << factors[i];
        }

        if (factors.empty()) oss << "1";

        oss << ")";
        return oss.str();
    }

    expr_t expr() const {
        if (factors.size() > 1 && jit::is_one(factors.back())) {
            std::vector<expr_t> f(factors.begin(), factors.end() - 1);
            return make_nary_op(op_kind_t::_mul, f);
        }
        return make_nary_op(op_kind_t::_mul, factors);
    }

    expr_t const_factor() const { return factors.back(); }

    bool is_one() const {
        return (factors.size() == 1) && jit::is_one(factors[0]);
    }

    bool is_const() const { return factors.size() == 1; }

    // Returns multiplication of this and other as factored_expr_t.
    expr_t merge(const expr_t &other) const {
        auto &f_other = other.as<factored_expr_t>();
        std::vector<expr_t> merged_factors(factors.begin(), factors.end());
        merged_factors.insert(merged_factors.end(), f_other.factors.begin(),
                f_other.factors.end());
        return factored_expr_t::make(type, merged_factors);
    }

    // Returns common factors of this and other as factored_expr_t.
    expr_t intersect(const expr_t &other) const {
        return intersect_impl(other, false);
    }

    // Returns common factors of this and other as factored_expr_t (ignores
    // constant factors).
    expr_t intersect_ignore_const(const expr_t &other) const {
        return intersect_impl(other, true);
    }

    // Returns factors of this not presented in other as factored_expr_t.
    expr_t diff(const expr_t &_other) const {
        auto &other = _other.as<factored_expr_t>();
        object_eq_map_t<expr_t, int> f_map;
        // Skip constant factor.
        for (size_t i = 0; i < factors.size() - 1; i++)
            f_map[factors[i]]++;

        for (auto &e : other.factors) {
            if (f_map[e] > 0) f_map[e]--;
        }
        std::vector<expr_t> diff_factors;
        for (auto &kv : f_map) {
            for (int i = 0; i < kv.second; i++)
                diff_factors.push_back(kv.first);
        }
        // Handle constant factor.
        int64_t a_const = to_cpp<int64_t>(factors.back());
        int64_t b_const = to_cpp<int64_t>(other.factors.back());
        if (a_const != 0 && b_const != 0) {
            int64_t ab_gcd = ((a_const < 0) && (b_const < 0)) ? -1 : 1;
            ab_gcd *= math::gcd(std::abs(a_const), std::abs(b_const));
            diff_factors.push_back(to_expr(a_const / ab_gcd, type));
        } else if (a_const != 0 || b_const != 0) {
            diff_factors.push_back(to_expr(a_const, type));
        }

        return factored_expr_t::make(type, diff_factors);
    }

    // Returns factors of this reduced by factors of other as factored_expr_t.
    // This object must be reducible by other.
    expr_t reduce(const expr_t &other) const {
        auto &f_other = other.as<factored_expr_t>();
        auto f_common = intersect(other);
        auto diff_other = f_other.diff(f_common);
        // Other must be reducible.
        ir_assert(diff_other.as<factored_expr_t>().is_one()) << diff_other;
        return diff(f_common);
    }

    // Returns true if this can be reduced by other.
    bool is_reducible(const expr_t &other) const {
        auto f_common = intersect(other);
        return f_common.is_equal(other);
    }

    static void reduce(expr_t &a, expr_t &b) {
        auto fa_expr = factored_expr_t::make(a);
        auto fb_expr = factored_expr_t::make(b);
        auto &fa = fa_expr.as<factored_expr_t>();
        auto &fb = fb_expr.as<factored_expr_t>();
        auto f_common = fa.intersect(&fb);
        a = fa.reduce(f_common).as<factored_expr_t>().expr();
        b = fb.reduce(f_common).as<factored_expr_t>().expr();
    }

    std::vector<expr_t> factors;

private:
    factored_expr_t(const expr_t &e) : expr_impl_t(e.type()) {
        init_factors(e);
    }

    factored_expr_t(const type_t &type, const std::vector<expr_t> &factors)
        : expr_impl_t(type) {
        init_normalize(factors);
    }

    void init_normalize(const std::vector<expr_t> &f) {
        bool sign = false;
        expr_t e_const = to_expr(1);
        for (auto &e : f) {
            if (!jit::is_const(e)) {
                factors.push_back(e);
                continue;
            }
            if (to_cpp<int64_t>(e) < 0) sign = !sign;
            if (jit::is_one(e) || jit::is_minus_one(e)) continue;

            e_const = e_const * abs(e);
        }
        if (sign) e_const = -e_const;
        factors.push_back(e_const);
    }

    void init_factors(const expr_t &e) {
        auto *nary = e.as_ptr<nary_op_t>();
        if (!nary) {
            init_normalize({e});
            return;
        }

        if (nary->op_kind == op_kind_t::_mul) {
            expr_t f_mul = factored_expr_t::make(to_expr(1));
            for (auto &a : nary->args) {
                f_mul = f_mul.as<factored_expr_t>().merge(
                        factored_expr_t::make(a));
            }
            factors = f_mul.as<factored_expr_t>().factors;
            return;
        }

        if (nary->op_kind == op_kind_t::_add) {
            expr_t common;
            for (auto &a : nary->args) {
                if (common.is_empty()) {
                    common = factored_expr_t::make(a);
                    continue;
                }
                common = common.as<factored_expr_t>().intersect(
                        factored_expr_t::make(a));
            }
            if (common.as<factored_expr_t>().is_one()) {
                init_normalize({e});
                return;
            }
            std::vector<expr_t> rest_factors;
            for (auto &a : nary->args) {
                auto fa_expr = factored_expr_t::make(a);
                auto &fa = fa_expr.as<factored_expr_t>();
                rest_factors.push_back(
                        fa.reduce(common).as<factored_expr_t>().expr());
            }
            auto &f_common = common.as<factored_expr_t>();
            auto rest = factored_expr_t::make(
                    make_nary_op(op_kind_t::_add, rest_factors));
            factors = f_common.merge(rest).as<factored_expr_t>().factors;
            return;
        }
        ir_error_not_expected();
    }

    expr_t intersect_impl(const expr_t &other, bool ignore_constants) const {
        auto &f_other = other.as<factored_expr_t>();
        object_eq_map_t<expr_t, int> f_map;
        // Skip constant factor.
        for (size_t i = 0; i < factors.size() - 1; i++)
            f_map[factors[i]]++;

        std::vector<expr_t> common_factors;
        for (auto &e : f_other.factors) {
            auto it = f_map.find(e);
            if (it == f_map.end() || it->second == 0) continue;
            f_map[e]--;
            common_factors.push_back(e);
        }

        if (ignore_constants)
            return factored_expr_t::make(type, common_factors);

        // Handle constant factor.
        int64_t a_const = to_cpp<int64_t>(factors.back());
        int64_t b_const = to_cpp<int64_t>(f_other.factors.back());
        if (a_const != 0 && b_const != 0) {
            int64_t ab_gcd = ((a_const < 0) && (b_const < 0)) ? -1 : 1;
            ab_gcd *= math::gcd(std::abs(a_const), std::abs(b_const));
            if (ab_gcd != 1) common_factors.push_back(to_expr(ab_gcd, type));
        } else if (a_const == 0 && b_const == 0) {
            common_factors.push_back(to_expr(0, type));
        }

        return factored_expr_t::make(type, common_factors);
    }
};

class division_reducer_t : public nary_op_mutator_t {
public:
    using nary_op_mutator_t::_mutate;

    object_t _mutate(const binary_op_t *obj) override {
        if (obj->op_kind != op_kind_t::_div)
            return nary_op_mutator_t::_mutate(obj);

        expr_t a = mutate(obj->a);
        expr_t b = mutate(obj->b);

        factored_expr_t::reduce(a, b);

        if (is_one(b)) return std::move(a);

        return binary_op_t::make(op_kind_t::_div, a, b);
    }
};

bool is_divisible(
        const expr_t &a, const expr_t &b, const constraint_set_t &cset) {
    if (cset.can_prove(a % b == 0, /*try_simplify=*/false)) return true;

    // Try to find b in factors of a.
    auto fa = factored_expr_t::make(a);
    auto fb = factored_expr_t::make(b);
    return fa.as<factored_expr_t>().is_reducible(fb);
}

class int_div_mod_expander_t : public nary_op_mutator_t {
public:
    using nary_op_mutator_t::_mutate;

    int_div_mod_expander_t(const constraint_set_t &cset) : cset(cset) {}

    object_t _mutate(const binary_op_t *_obj) override {
        auto obj = nary_op_mutator_t::_mutate(_obj);
        auto *binary_op = obj.as_ptr<binary_op_t>();
        if (!binary_op) return obj;
        if (!utils::one_of(
                    binary_op->op_kind, op_kind_t::_div, op_kind_t::_mod))
            return obj;
        if (!binary_op->type.is_int()) return obj;

        auto a = binary_op->a;
        auto b = binary_op->b;

        auto _b = nary_op_back_transform(b);
        if (!cset.can_prove(_b > 0)) return obj;

        auto *a_nary = a.as_ptr<nary_op_t>();

        if (a_nary && a_nary->op_kind == op_kind_t::_add)
            return mutate_with_add(binary_op);

        // Try to reduce a and b.
        factored_expr_t::reduce(a, b);

        if (is_one(b)) {
            if (binary_op->op_kind == op_kind_t::_mod)
                return to_expr(0, binary_op->type);
            if (binary_op->op_kind == op_kind_t::_div) return std::move(a);
        }

        if (binary_op->op_kind == op_kind_t::_div) return a / b;

        return obj;
    }

    expr_t mutate_with_add(const binary_op_t *obj) {
        expr_t e = obj;
        if (reduce_v1(e)) return e;
        if (reduce_v2(e)) return e;
        return e;
    }

    // Applies the following rules:
    // 1) (A + B) % C -> B % C, when
    //     - A % C == 0
    //     - B >= 0
    // 2) (A + B) / C -> (A / C) + (B / C), when
    //     - A % C == 0
    //     - B >= 0
    bool reduce_v1(expr_t &expr) {
        auto *binary_op = expr.as_ptr<binary_op_t>();
        if (!binary_op) return false;

        auto op_kind = binary_op->op_kind;
        auto &a = binary_op->a;
        auto &b = binary_op->b;

        std::vector<expr_t> lhs_args; // Reducible summands.
        std::vector<expr_t> rhs_args; // Non-reducible summands.

        auto *a_nary = a.as_ptr<nary_op_t>();
        for (auto &e : a_nary->args) {
            if (is_div_reducible(e, b)) {
                lhs_args.push_back(e);
            } else {
                rhs_args.push_back(e);
            }
        }

        // Nothing to reduce, return expression as is.
        if (lhs_args.empty()) return false;

        auto rhs_nary = make_nary_op(op_kind_t::_add, rhs_args);
        auto _rhs = nary_op_back_transform(rhs_nary);
        bool rhs_ge_0 = cset.can_prove(_rhs >= 0);

        if (op_kind == op_kind_t::_mod) {
            if (rhs_args.empty()) {
                expr = to_expr(0, expr.type());
                return true;
            }
            if (!rhs_ge_0) return false;
            expr = rhs_nary % b;
            return true;
        }

        if (op_kind == op_kind_t::_div) {
            if (!rhs_ge_0) return false;
            if (rhs_args.empty()) {
                expr = mutate(lhs_args[0] / b);
                for (int i = 1; i < int(lhs_args.size()); i++) {
                    expr += mutate(lhs_args[i] / b);
                }
                return true;
            }
            auto lhs_div = make_nary_op(op_kind_t::_add, lhs_args) / b;
            auto rhs_div = rhs_nary / b;
            expr = mutate(lhs_div) + mutate(rhs_div);
            return true;
        }

        ir_error_not_expected() << expr;

        return false;
    }

    // Applies the following rules:
    // 1) (A * B + D) / (A * C) -> (A * B) / (A * C), when
    //     - A > 0
    //     - C > 0
    //     - 0 <= D < A
    // 2) (A * B + D) % (A * C) -> (A * B) % (A * C) + D % (A * C), when
    //     - A > 0
    //     - C > 0
    //     - 0 <= D < A
    bool reduce_v2(expr_t &expr) {
        auto *binary_op = expr.as_ptr<binary_op_t>();
        if (!binary_op) return false;

        auto op_kind = binary_op->op_kind;
        auto &a = binary_op->a;
        auto &b = binary_op->b;
        if (!is_const(b)) return false;

        auto const_factor = [&](const expr_t &e) {
            auto _fe = factored_expr_t::make(e);
            auto &fe = _fe.as<factored_expr_t>();
            auto ret = to_cpp<int64_t>(fe.const_factor());
            for (auto &f : fe.factors)
                if (is_var(f)) ret *= cset.max_proven_gcd(f);
            return ret;
        };

        // TODO: Check 0.
        // Find max constant GCD.
        int64_t b_gcd = const_factor(b);
        int64_t max_gcd = 0;
        auto *a_nary = a.as_ptr<nary_op_t>();
        for (auto &e : a_nary->args) {
            int64_t gcd = math::gcd(b_gcd, const_factor(e));
            if (gcd > max_gcd) max_gcd = gcd;
        }

        if (max_gcd == 0) return false;

        std::vector<expr_t> lhs_args; // Reducible summands.
        std::vector<expr_t> rhs_args; // Non-reducible summands.
        for (auto &e : a_nary->args) {
            if (is_div_reducible(e, max_gcd)) {
                lhs_args.push_back(e);
            } else {
                rhs_args.push_back(e);
            }
        }

        // max_gcd is the GCD for some summand so at least one summand must be
        // reducible.
        ir_assert(!lhs_args.empty());

        if (rhs_args.empty()) return false;

        int64_t A = max_gcd;
        int64_t C = to_cpp<int64_t>(b) / A;
        if (A <= 0 || C <= 0) return false;

        auto rhs_nary = make_nary_op(op_kind_t::_add, rhs_args);
        auto D = nary_op_back_transform(rhs_nary);
        if (!cset.can_prove(D >= 0) || !cset.can_prove(D < A)) return false;

        if (op_kind == op_kind_t::_mod) {
            auto lhs_mod = make_nary_op(op_kind_t::_add, lhs_args) % b;
            auto rhs_mod = rhs_nary % b;
            expr = mutate(lhs_mod) + mutate(rhs_mod);
            return true;
        }

        if (op_kind == op_kind_t::_div) {
            auto lhs_div = make_nary_op(op_kind_t::_add, lhs_args) / b;
            expr = lhs_div;
            return true;
        }

        ir_error_not_expected() << expr;

        return false;
    }

    bool is_div_reducible(const expr_t &a, const expr_t &b) const {
        if (is_const(a) && is_const(b)) {
            return to_cpp<int64_t>(a) % to_cpp<int64_t>(b) == 0;
        }

        if (b.is_equal(to_expr(1, b.type()))) return true;

        return is_divisible(a, b, cset);
    }

    const constraint_set_t &cset;
};

class int_div_mod_range_simplifier_t : public nary_op_mutator_t {
public:
    using nary_op_mutator_t::_mutate;

    int_div_mod_range_simplifier_t(const constraint_set_t &cset) : cset(cset) {}

    object_t _mutate(const binary_op_t *obj) override {
        if (!utils::one_of(obj->op_kind, op_kind_t::_div, op_kind_t::_mod))
            return nary_op_mutator_t::_mutate(obj);

        auto a = mutate(obj->a);
        auto b = mutate(obj->b);

        auto _a = nary_op_back_transform(a);
        auto _b = nary_op_back_transform(b);

        // 0 <= a < b => (a / b) == 0
        bool abs_a_lt_b = cset.can_prove(_a >= 0) && cset.can_prove(_a < _b);

        // 0 <= a < b => (a % b) == a
        if (abs_a_lt_b) {
            if (obj->op_kind == op_kind_t::_div) return to_expr(0);
            if (obj->op_kind == op_kind_t::_mod) return a;
        }

        return binary_op_t::make(obj->op_kind, a, b);
    }

    const constraint_set_t &cset;
};

// Factors out common factors in an N-ary expression.
class common_factor_simplifier_t : public nary_op_mutator_t {
public:
    object_t _mutate(const nary_op_t *obj) override {
        if (obj->op_kind != op_kind_t::_add)
            return nary_op_mutator_t::_mutate(obj);

        auto args = mutate(obj->args);
        for (auto &a : args) {
            auto *nary = a.as_ptr<nary_op_t>();
            if (nary) ir_assert(nary->op_kind == op_kind_t::_mul) << a;
        }

        // Fold same factors (find exact match, ignore constants).
        // Example:
        //     (a * c1 + a * c2 + b) ->
        //     (a * c3 + b) where c3 = (c1 + c2)
        for (size_t i = 0; i < args.size(); i++) {
            auto e_fi = factored_expr_t::make(args[i]);
            for (size_t j = i + 1; j < args.size(); j++) {
                auto e_fj = factored_expr_t::make(args[j]);

                auto &fi = e_fi.as<factored_expr_t>();
                auto &fj = e_fj.as<factored_expr_t>();

                auto e_fij_common = fi.intersect_ignore_const(e_fj);
                auto &fij_common = e_fij_common.as<factored_expr_t>();
                if (fi.is_equal_ignore_const(&fij_common)
                        && fj.is_equal_ignore_const(&fij_common)) {
                    auto new_args = fij_common.factors;
                    new_args.push_back(fi.const_factor() + fj.const_factor());
                    args[i] = make_nary_op(op_kind_t::_mul, new_args);
                    e_fi = factored_expr_t::make(args[i]);
                    args[j] = to_expr(0, args[j].type());
                }
            }
        }

        // Partial folding (fold any match).
        // Example:
        //     (a * b * c + a * b * d + e) ->
        //     ((a * b * (c + d)) + e)
        for (size_t i = 0; i < args.size(); i++) {
            if (is_zero(args[i])) continue;
            auto e_fi = factored_expr_t::make(args[i]);
            for (size_t j = i + 1; j < args.size(); j++) {
                if (is_zero(args[j])) continue;
                auto e_fj = factored_expr_t::make(args[j]);

                auto &fi = e_fi.as<factored_expr_t>();
                auto &fj = e_fj.as<factored_expr_t>();

                auto e_fij_common = fi.intersect_ignore_const(e_fj);
                auto &fij_common = e_fij_common.as<factored_expr_t>();

                // fij_common = 1 means no common factors, other constant
                // factors are also ignored, for simplicity (though it might be
                // beneficial to fold them as well).
                if (fij_common.is_const()) continue;

                // factored_expr_t::make() will find common factors.
                auto e_fi_add_fj = factored_expr_t::make(
                        make_nary_op(op_kind_t::_add, {fi.expr(), fj.expr()}));
                auto &fi_add_fj = e_fi_add_fj.as<factored_expr_t>();
                args[i] = make_nary_op(op_kind_t::_mul, fi_add_fj.factors);
                e_fi = e_fi_add_fj;
                args[j] = to_expr(0, args[j].type());
            }
        }

        return make_nary_op(obj->op_kind, args);
    }
};

// Simplifies using the N-ary form.
expr_t simplify_with_nary(const expr_t &_e, const constraint_set_t &cset) {
    auto e = _e;

    if (!e.type().is_scalar() || e.type().is_fp()) { return e; }
    e = nary_op_canonicalize(e);

    e = division_reducer_t().mutate(e);
    e = nary_op_flatten(e);
    e = int_div_mod_expander_t(cset).mutate(e);
    e = common_factor_simplifier_t().mutate(e);
    e = int_div_mod_range_simplifier_t(cset).mutate(e);

    e = nary_op_back_transform(e);

    return e;
}

class stmt_simplifier_t : public ir_mutator_t {
public:
    stmt_simplifier_t(const constraint_set_t &cset) : cset_(cset) {}

    object_t _mutate(const binary_op_t *obj) override {
        return simplify(obj, cset_);
    }

    object_t _mutate(const if_t *obj) override {
        auto cond = simplify(obj->cond);

        if (is_const(cond)) {
            if (to_cpp<bool>(cond)) return mutate(obj->body);
            return mutate(obj->else_body);
        }

        auto body = obj->body;
        if (!body.is_empty()) {
            auto cset_old = cset_;
            cset_.add_constraint(cond);
            body = ir_mutator_t::mutate(body);
            cset_ = cset_old;
        }

        auto else_body = obj->else_body;
        if (!else_body.is_empty()) {
            auto cset_old = cset_;
            cset_.add_constraint(flip_condition(cond));
            else_body = ir_mutator_t::mutate(else_body);
            cset_ = cset_old;
        }

        return if_t::make(cond, body, else_body);
    }

    object_t _mutate(const let_t *obj) override {
        // External variable.
        if (obj->value.is_empty()) return ir_mutator_t::_mutate(obj);

        // Substitute constants.
        auto value = simplify(obj->value);
        if (is_const(value)) {
            auto body = substitute(obj->body, obj->var, value);
            return mutate(body);
        }
        auto cset_old = cset_;
        cset_.add_constraint(obj->var == value);
        auto new_obj = let_t::make(obj->var, value, obj->body);
        new_obj = ir_mutator_t::_mutate(new_obj.as_ptr<let_t>());
        cset_ = cset_old;

        return std::move(new_obj);
    }

    object_t _mutate(const for_t *obj) override {
        if (is_zero(obj->init) && is_one(obj->bound)) {
            auto body = substitute(obj->body, obj->var, expr_t(0));
            return mutate(body);
        }

        auto cset_old = cset_;
        cset_.add_constraint(obj->var >= obj->init);
        cset_.add_constraint(obj->var < obj->bound);
        auto new_obj = ir_mutator_t::_mutate(obj);
        cset_ = cset_old;

        return new_obj;
    }

private:
    static op_kind_t flip_cmp_op(op_kind_t op_kind) {
        switch (op_kind) {
            case op_kind_t::_eq: return op_kind_t::_ne;
            case op_kind_t::_ge: return op_kind_t::_lt;
            case op_kind_t::_gt: return op_kind_t::_le;
            case op_kind_t::_le: return op_kind_t::_gt;
            case op_kind_t::_lt: return op_kind_t::_ge;
            case op_kind_t::_ne: return op_kind_t::_eq;
            default: ir_error_not_expected();
        }
        return op_kind_t::undef;
    }

    static expr_t flip_condition(const expr_t &cond) {
        ir_assert(cond.type().is_bool());

        auto *binary_op = cond.as_ptr<binary_op_t>();
        if (binary_op) {
            auto &a = binary_op->a;
            auto &b = binary_op->b;
            auto op_kind = binary_op->op_kind;
            return binary_op_t::make(flip_cmp_op(op_kind), a, b);
        }

        auto *shuffle = cond.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast()) {
            return shuffle_t::make_broadcast(
                    flip_condition(shuffle->vec[0]), shuffle->elems());
        }

        ir_error_not_expected();
        return expr_t();
    }

    constraint_set_t cset_;
};

expr_t simplify_expr(const expr_t &_e, const constraint_set_t &cset) {
    expr_t e = _e;

    if (is_const(e) || is_var(e)) return e;

    e = const_fold(e);
    e = simplify_rewrite(e);

    e = simplify_comparison(e);
    e = range_simplifier_t(cset).mutate(e);
    e = simplify_with_nary(e, cset);

    e = const_fold(e);
    e = simplify_rewrite(e);

    return e;
}

stmt_t simplify_stmt(const stmt_t &s, const constraint_set_t &cset) {
    stmt_simplifier_t simplifier(cset);
    return simplifier.mutate(s);
}

template <op_kind_t op_kind>
struct op_traits_t {};

#define DECL_OP_TRAITS(name, op) \
    template <> \
    struct op_traits_t<name> { \
        template <typename T, \
                typename = typename std::enable_if< \
                        !std::is_same<T, bool>::value>::type> \
        static auto compute(T a, T b) -> decltype(a op b) { \
            return a op b; \
        } \
        template <op_kind_t dummy_op = name, \
                typename \
                = typename std::enable_if<dummy_op == op_kind_t::_and>::type> \
        static bool compute(bool a, bool b) { \
            return a op b; \
        } \
    };

DECL_OP_TRAITS(op_kind_t::_add, +)
DECL_OP_TRAITS(op_kind_t::_sub, -)
DECL_OP_TRAITS(op_kind_t::_mul, *)
DECL_OP_TRAITS(op_kind_t::_div, /)
DECL_OP_TRAITS(op_kind_t::_mod, %)

DECL_OP_TRAITS(op_kind_t::_eq, ==)
DECL_OP_TRAITS(op_kind_t::_ne, !=)
DECL_OP_TRAITS(op_kind_t::_gt, >)
DECL_OP_TRAITS(op_kind_t::_ge, >=)
DECL_OP_TRAITS(op_kind_t::_lt, <)
DECL_OP_TRAITS(op_kind_t::_le, <=)

DECL_OP_TRAITS(op_kind_t::_and, &&)

#undef DECL_OP_TRAITS

template <op_kind_t op_kind, typename T, typename = void>
struct compute_helper_t {
    static expr_t call(T a, T b) { return expr_t(); }
};

template <typename>
struct voider_t {
    using type = void;
};

template <op_kind_t op_kind, typename T>
struct compute_helper_t<op_kind, T,
        typename voider_t<decltype(
                op_traits_t<op_kind>::compute(T(), T()))>::type> {
    static expr_t call(T a, T b) {
        return to_expr(op_traits_t<op_kind>::compute(a, b));
    }
};

template <typename T>
class const_fold_helper_t {
public:
    template <typename U = T>
    static expr_t call(op_kind_t op_kind, T a, T b) {
        switch (op_kind) {
#define CASE(op) \
    case op: return compute_helper_t<op, T>::call(a, b);

            CASE(op_kind_t::_add)
            CASE(op_kind_t::_sub)
            CASE(op_kind_t::_mul)
            CASE(op_kind_t::_div)
            CASE(op_kind_t::_mod)

            CASE(op_kind_t::_eq)
            CASE(op_kind_t::_ne)
            CASE(op_kind_t::_gt)
            CASE(op_kind_t::_ge)
            CASE(op_kind_t::_lt)
            CASE(op_kind_t::_le)

            CASE(op_kind_t::_and)

            default: ir_error_not_expected();

#undef CASE
        }
        return expr_t();
    }
};

class const_folder_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t *obj) override {
        return mutate_expr(obj);
    }
    object_t _mutate(const cast_t *obj) override { return mutate_expr(obj); }
    object_t _mutate(const iif_t *obj) override { return mutate_expr(obj); }
    object_t _mutate(const unary_op_t *obj) override {
        return mutate_expr(obj);
    }

private:
    template <typename T>
    object_t mutate_expr(const T *obj) {
        auto new_obj = ir_mutator_t::_mutate(obj);
        return const_fold_non_recursive(new_obj);
    }
};

bool is_const_or_shuffle_const(const expr_t &e) {
    return is_const(e) || is_shuffle_const(e);
}

expr_t const_fold_unary(op_kind_t op_kind, const expr_t &a) {
    ir_assert(op_kind == op_kind_t::_minus);
    if (!a.type().is_scalar()) {
        int elems = a.type().elems();
        std::vector<expr_t> ret;
        for (int i = 0; i < elems; i++) {
            ret.push_back(const_fold_unary(op_kind, a[i]));
        }
        return shuffle_t::make(ret);
    }

#define CASE(ir_type, cpp_type) \
    if (a.type() == type_t::ir_type()) return to_expr(-to_cpp<cpp_type>(a))

    CASE(f32, float);
    CASE(s16, int16_t);
    CASE(s32, int32_t);
    CASE(s64, int64_t);

    if (a.type().is_bool()) return to_expr(!to_cpp<bool>(a));

#undef CASE

    ir_error_not_expected() << "Cannot handle type: " << a;
    return expr_t();
}

expr_t const_fold_binary(const type_t &compute_type, op_kind_t op_kind,
        const expr_t &a, const expr_t &b) {
    if (!compute_type.is_scalar()) {
        int elems = compute_type.elems();
        auto scalar_type = compute_type.scalar();
        std::vector<expr_t> ret;
        for (int i = 0; i < elems; i++) {
            ret.push_back(const_fold_binary(scalar_type, op_kind, a[i], b[i]));
        }
        return shuffle_t::make(ret);
    }

    if (compute_type.is_unsigned()) {
        auto a_s64 = to_cpp<int64_t>(a);
        auto b_s64 = to_cpp<int64_t>(b);
        ir_assert(a_s64 >= 0 && b_s64 >= 0)
                << "Overflow detected: fix data types.";
        MAYBE_UNUSED(a_s64);
        MAYBE_UNUSED(b_s64);
    }

#define CASE(ir_type, cpp_type) \
    if (compute_type == type_t::ir_type()) { \
        auto _a = to_cpp<cpp_type>(a); \
        auto _b = to_cpp<cpp_type>(b); \
        return const_fold_helper_t<cpp_type>::call(op_kind, _a, _b); \
    }

    CASE(_bool, bool)
    CASE(f32, float)
    CASE(s16, int16_t)
    CASE(s32, int32_t)
    CASE(s64, int64_t)
    CASE(u16, uint16_t)
    CASE(u32, uint32_t)
    CASE(u64, uint64_t)

#undef CASE

    ir_error_not_expected() << "Unknown type.";
    return expr_t();
}

object_t simplify(const object_t &obj, const constraint_set_t &cset) {
    if (obj.is_expr()) return simplify_expr(obj, cset);
    if (obj.is_stmt()) return simplify_stmt(obj, cset);
    ir_assert(obj.is_empty());
    return object_t();
}

expr_t simplify_cmp_move_const_to_rhs(const expr_t &e) {
    if (!is_binary_cmp_op(e)) return e;

    auto &op = e.as<binary_op_t>();
    if (!is_const(op.b)) return e;
    if (!is_binary_op(op.a)) return e;

    auto &a_op = op.a.as<binary_op_t>();

    bool is_lhs_add = (a_op.op_kind == op_kind_t::_add);
    bool is_lhs_sub = (a_op.op_kind == op_kind_t::_sub);
    if (!is_lhs_add && !is_lhs_sub) return e;

    auto &c1 = op.b;

    expr_t lhs;
    expr_t rhs;
    op_kind_t op_kind;
    if (is_const(a_op.a)) {
        auto &c0 = a_op.a;
        auto &x = a_op.b;
        if (is_lhs_add) {
            // ((c0 + x) op c1) ->  (x op (c1 - c0))
            lhs = x;
            rhs = c1 - c0;
            op_kind = op.op_kind;
        } else {
            // ((c0 - x) op c1) ->  (x -op (c0 - c1))
            lhs = x;
            rhs = c0 - c1;
            op_kind = negate_cmp_op(op.op_kind);
        }
    } else if (is_const(a_op.b)) {
        auto &x = a_op.a;
        auto &c0 = a_op.b;
        if (is_lhs_add) {
            // ((x + c0) op c1) ->  (x op (c1 - c0))
            lhs = x;
            rhs = c1 - c0;
            op_kind = op.op_kind;
        } else {
            // ((x - c0) op c1) ->  (x op (c0 + c1))
            lhs = x;
            rhs = c0 + c1;
            op_kind = op.op_kind;
        }
    } else {
        return e;
    }
    return binary_op_t::make(op_kind, lhs, rhs);
}

expr_t simplify_cmp_reduce_lhs_rhs(const expr_t &e) {
    if (!is_binary_cmp_op(e)) return e;

    auto &op = e.as<binary_op_t>();

    // Rule:
    //     (c0 * x op c1) or (x * c0 op c1) ->
    //     x new_op (c1 / c0) if abs(c1) % abs(c0) == 0
    //     new_op == op or new_op == negate_cmp_op(op)
    expr_t c0;
    expr_t c1 = op.b;
    expr_t x;

    if (!is_const(c1)) return e;
    if (!is_binary_op(op.a, op_kind_t::_mul)) return e;

    auto &a_op = op.a.as<binary_op_t>();
    if (is_const(a_op.a)) {
        c0 = a_op.a;
        x = a_op.b;
    } else if (is_const(a_op.b)) {
        x = a_op.a;
        c0 = a_op.b;
    }

    if (c0.is_empty()) return e;
    if (!c0.type().is_int()) return e;
    if (!c1.type().is_int()) return e;

    auto i_c0 = to_cpp<int64_t>(c0);
    auto i_c1 = to_cpp<int64_t>(c1);

    bool is_c0_neg = (i_c0 < 0);
    bool sign = ((i_c0 < 0) != (i_c1 < 0));
    i_c0 = std::abs(i_c0);
    i_c1 = std::abs(i_c1);

    bool has_mod = (i_c1 % i_c0 != 0);
    if (has_mod && utils::one_of(op.op_kind, op_kind_t::_eq, op_kind_t::_ne))
        return e;

    auto new_op_kind = (is_c0_neg ? negate_cmp_op(op.op_kind) : op.op_kind);
    int64_t div = i_c1 / i_c0;
    if (has_mod) {
        switch (new_op_kind) {
            case op_kind_t::_ge:
            case op_kind_t::_gt:
                new_op_kind = op_kind_t::_ge;
                div = (sign ? div : div + 1);
                break;
            case op_kind_t::_le:
            case op_kind_t::_lt:
                new_op_kind = op_kind_t::_le;
                div = (sign ? div + 1 : div);
                break;
            default: ir_error_not_expected();
        }
    }

    return binary_op_t::make(new_op_kind, x, (sign ? -1 : 1) * div);
}

bool const_to_const_binary(const expr_t &e, op_kind_t op_kind,
        const type_t &a_type, const type_t &b_type, expr_t &a, expr_t &b) {
    bool is_true = to_cpp<bool>(e);
    // Assume:
    // - a0 < b1
    // - a1 > b0
    // - a_eq == b_eq
    expr_t a0 = to_expr(0, a_type);
    expr_t a1 = to_expr(1, a_type);
    expr_t b0 = to_expr(0, b_type);
    expr_t b1 = to_expr(1, b_type);
    expr_t a_eq = to_expr(0, a_type);
    expr_t b_eq = to_expr(0, b_type);
    if (!a.is_empty()) {
        a0 = a1 = a;
        b0 = a - 1;
        b1 = a + 1;
        a_eq = b_eq = a;
    } else if (!b.is_empty()) {
        b0 = b1 = b;
        a0 = b - 1;
        a1 = b + 1;
        a_eq = b_eq = b;
    }
    switch (op_kind) {
        case op_kind_t::_and: a = b = e; return true;
        case op_kind_t::_le:
        case op_kind_t::_lt:
            a = (is_true ? a0 : a1);
            b = (is_true ? b1 : b0);
            return true;
        case op_kind_t::_ge:
        case op_kind_t::_gt:
            a = (is_true ? a1 : a0);
            b = (is_true ? b0 : b1);
            return true;
        case op_kind_t::_eq:
            a = (is_true ? a_eq : a0);
            b = (is_true ? b_eq : b1);
            return true;
        case op_kind_t::_ne:
            a = (is_true ? a0 : a_eq);
            b = (is_true ? b1 : b_eq);
            return true;
        default: return false;
    }
}

expr_t simplify_propagate_shuffle(const expr_t &e) {
    if (!e.type().is_bool()) return e;

    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return e;

    // Handle binary operation.
    {
        type_t a_type;
        type_t b_type;
        expr_t a_common_const;
        expr_t b_common_const;
        op_kind_t op_kind = op_kind_t::undef;
        bool found_binary = false;
        for (int i : shuffle->idx) {
            if (is_binary_op(shuffle->vec[i])) {
                found_binary = true;
                auto &op = shuffle->vec[i].as<binary_op_t>();
                a_type = op.a.type();
                b_type = op.b.type();
                op_kind = op.op_kind;
                if (is_const(op.a)) a_common_const = op.a;
                if (is_const(op.b)) b_common_const = op.b;
                break;
            }
        }
        if (!found_binary) return e;

        for (int i : shuffle->idx) {
            auto &elem = shuffle->vec[i];
            if (is_binary_op(elem, op_kind)) {
                auto &op = elem.as<binary_op_t>();
                if (!a_common_const.is_equal(op.a)) {
                    a_common_const = expr_t();
                }
                if (!b_common_const.is_equal(op.b)) {
                    b_common_const = expr_t();
                }
            }
        }

        bool ok = true;
        std::vector<expr_t> a;
        std::vector<expr_t> b;
        for (int i : shuffle->idx) {
            auto &elem = shuffle->vec[i];
            if (is_binary_op(elem, op_kind)) {
                auto &op = elem.as<binary_op_t>();
                a.push_back(op.a);
                b.push_back(op.b);
                continue;
            }
            if (is_const(elem)) {
                expr_t op_a = a_common_const;
                expr_t op_b = b_common_const;
                if (!const_to_const_binary(
                            elem, op_kind, a_type, b_type, op_a, op_b)) {
                    ok = false;
                    break;
                }
                a.push_back(op_a);
                b.push_back(op_b);
                continue;
            }
            ok = false;
            break;
        }
        if (ok) {
            auto _a = simplify_propagate_shuffle(shuffle_t::make(a));
            auto _b = simplify_propagate_shuffle(shuffle_t::make(b));
            return binary_op_t::make(op_kind, _a, _b);
        }
    }

    return e;
}

expr_t const_fold_non_recursive(const expr_t &e) {
    auto *unary_op = e.as_ptr<unary_op_t>();
    if (unary_op) {
        auto &a = unary_op->a;
        if (!is_const_or_shuffle_const(a)) return e;
        return const_fold_unary(unary_op->op_kind, a);
    }

    auto *binary_op = e.as_ptr<binary_op_t>();
    if (binary_op) {
        auto op_kind = binary_op->op_kind;
        auto &a = binary_op->a;
        auto &b = binary_op->b;
        if (!is_const_or_shuffle_const(a) || !is_const_or_shuffle_const(b))
            return e;

        auto compute_type = common_type(a, b);
        return const_fold_binary(compute_type, op_kind, a, b);
    }

    auto *iif = e.as_ptr<iif_t>();
    if (iif) {
        if (!is_const(iif->cond)) return e;
        if (to_cpp<bool>(iif->cond)) return iif->true_expr;
        return iif->false_expr;
    }

    return e;
}

object_t const_fold(const object_t &obj) {
    return const_folder_t().mutate(obj);
}

expr_t nary_op_back_transform(const expr_t &e) {
    // Convert nary_op_t back to binary_op_t.
    return nary_op_back_transformer_t().mutate(e);
}

expr_t nary_op_canonicalize(const expr_t &_e) {
    auto e = _e;

    e = nary_op_transformer_t().mutate(e);
    e = nary_op_flatten(e);
    e = mul_nary_op_expander_t().mutate(e);

    ir_assert(is_nary_op_canonical(e)) << e;
    MAYBE_UNUSED(is_nary_op_canonical);

    return e;
}

expr_t make_nary_op(op_kind_t op_kind, const std::vector<expr_t> &args) {
    if (args.empty()) {
        if (op_kind == op_kind_t::_add) return 0;
        if (op_kind == op_kind_t::_mul) return 1;
        ir_error_not_expected() << to_string(op_kind);
    }
    if (args.size() == 1) return args[0];

    // Do eager constant folding.
    std::vector<expr_t> new_args;
    fold_const_nary_op_args(op_kind, args, new_args);

    if (new_args.size() < args.size()) return make_nary_op(op_kind, new_args);

    return nary_op_t::make(op_kind, new_args);
}

std::vector<expr_t> cvt_expr_to_nary_op_args(const expr_t &e) {
    auto *nary = e.as_ptr<nary_op_t>();
    if (nary) return nary->args;
    return {e};
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
