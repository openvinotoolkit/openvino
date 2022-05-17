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

#ifndef GPU_JIT_CONV_IR_HPP
#define GPU_JIT_CONV_IR_HPP

#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

#include "gpu/jit/conv/ir_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper class to walk through IR tree.
class ir_visitor_t {
public:
    using dispatch_func_type = void (*)(ir_visitor_t *, const object_impl_t *);

    virtual ~ir_visitor_t() = default;

    virtual void visit(const object_t &obj) { dispatch(obj.impl()); }

    template <typename T>
    void visit(const std::vector<T> &v) {
        for (auto &e : v)
            visit(e);
    }

    virtual void pre_visit(const object_impl_t *obj) {}
    virtual void post_visit(const object_impl_t *obj) {}

    // To catch missing _visit() handlers in ir_visitor_t.
    virtual void _visit(const object_impl_t *obj) {
        ir_error_not_expected() << "Can't handle type: " << object_t(obj);
    }

#define DECL_VISIT_LEAF(name) \
    virtual void _visit(const name *obj) {}

    DECL_VISIT_LEAF(bool_imm_t)
    DECL_VISIT_LEAF(float_imm_t)
    DECL_VISIT_LEAF(func_impl_t)
    DECL_VISIT_LEAF(int_imm_t)
    DECL_VISIT_LEAF(var_t)

#undef DECL_VISIT_LEAF

    virtual void _visit(const alloc_t *obj) {
        visit(obj->buf);
        visit(obj->body);
    }

    virtual void _visit(const binary_op_t *obj) {
        visit(obj->a);
        visit(obj->b);
    }

    virtual void _visit(const cast_t *obj) { visit(obj->expr); }

    virtual void _visit(const for_t *obj) {
        visit(obj->var);
        visit(obj->init);
        visit(obj->bound);
        visit(obj->body);
    }

    virtual void _visit(const func_call_t *obj) {
        visit(obj->func);
        visit(obj->args);
    }

    virtual void _visit(const if_t *obj) {
        visit(obj->cond);
        visit(obj->body);
        visit(obj->else_body);
    }

    virtual void _visit(const iif_t *obj) {
        visit(obj->cond);
        visit(obj->true_expr);
        visit(obj->false_expr);
    }

    virtual void _visit(const let_t *obj) {
        visit(obj->var);
        visit(obj->value);
        visit(obj->body);
    }

    virtual void _visit(const load_t *obj) {
        visit(obj->buf);
        visit(obj->off);
    }

    virtual void _visit(const ptr_t *obj) {
        visit(obj->base);
        visit(obj->off);
    }

    virtual void _visit(const shuffle_t *obj) { visit(obj->vec); }

    virtual void _visit(const stmt_group_t *obj) { visit(obj->body); }

    virtual void _visit(const stmt_seq_t *obj) {
        visit(obj->head);
        visit(obj->tail);
    }

    virtual void _visit(const store_t *obj) {
        visit(obj->buf);
        visit(obj->off);
        visit(obj->value);
        visit(obj->mask);
    }

    virtual void _visit(const ternary_op_t *obj) {
        visit(obj->a);
        visit(obj->b);
        visit(obj->c);
    }

    virtual void _visit(const unary_op_t *obj) { visit(obj->a); }

    bool is_supported(const object_t &obj) const {
        if (obj.is_empty()) return true;

        auto *impl = obj.impl();
        auto ti = impl->dispatch_type_id();
        return ti < num_dispatch_funcs;
    }

protected:
    virtual dispatch_func_type find_dispatch_func(int64_t ti) const {
        return ti < num_dispatch_funcs ? dispatch_funcs()[ti] : nullptr;
    }

private:
    static const int64_t num_dispatch_funcs
            = ir_type_id_t::end_visitable_ir_objects;
    static std::array<dispatch_func_type, num_dispatch_funcs> &
    dispatch_funcs() {
        static std::array<dispatch_func_type, num_dispatch_funcs>
                _dispatch_funcs;
        static std::once_flag initialized;
        std::call_once(initialized, [&]() {
#define HANDLE_IR_OBJECT(type) \
    _dispatch_funcs[type::_dispatch_type_id()] = &call<type>;
            HANDLE_ALL_IR_OBJECTS()

#undef HANDLE_IR_OBJECT
        });
        return _dispatch_funcs;
    }

    template <typename T>
    static void call(ir_visitor_t *visitor, const object_impl_t *obj) {
        visitor->pre_visit(obj);
        visitor->_visit((const T *)obj);
        visitor->post_visit(obj);
    }

    void dispatch(const object_impl_t *obj) {
        if (!obj) return;

        auto ti = obj->dispatch_type_id();
        auto f = find_dispatch_func(ti);
        if (!f) {
            ir_error_not_expected() << "Can't handle type: " << object_t(obj);
        }
        f(this, obj);
    }
};

// Helper class to mutate IR tree.
class ir_mutator_t {
public:
    using dispatch_func_type
            = object_t (*)(ir_mutator_t *, const object_impl_t *);

    virtual ~ir_mutator_t() = default;

    virtual object_t mutate(const object_t &obj) {
        return dispatch(obj.impl());
    }

    template <typename T>
    std::vector<T> mutate(const std::vector<T> &v) {
        std::vector<T> new_v;
        for (auto &e : v)
            new_v.push_back(mutate(e));
        return new_v;
    }

    // To catch missing _mutate() handlers ir ir_mutator_t.
    object_t _mutate(const object_impl_t *obj) {
        ir_error_not_expected() << "Can't handle type: " << object_t(obj);
        return {};
    }

#define DECL_MUTATE_LEAF(name) \
    virtual object_t _mutate(const name *obj) { return obj; }

    DECL_MUTATE_LEAF(bool_imm_t)
    DECL_MUTATE_LEAF(float_imm_t)
    DECL_MUTATE_LEAF(func_impl_t)
    DECL_MUTATE_LEAF(int_imm_t)
    DECL_MUTATE_LEAF(var_t)

#undef DECL_MUTATE_LEAF

    virtual object_t _mutate(const alloc_t *obj) {
        auto buf = mutate(obj->buf);
        auto body = mutate(obj->body);

        if (buf.is_same(obj->buf) && body.is_same(obj->body)) return obj;

        return alloc_t::make(buf, obj->size, obj->kind, obj->attr, body);
    }

    virtual object_t _mutate(const binary_op_t *obj) {
        auto a = mutate(obj->a);
        auto b = mutate(obj->b);

        if (a.is_same(obj->a) && b.is_same(obj->b)) return obj;

        return binary_op_t::make(obj->op_kind, a, b);
    }

    virtual object_t _mutate(const cast_t *obj) {
        auto expr = mutate(obj->expr);

        if (expr.is_same(obj->expr)) return obj;

        return cast_t::make(obj->type, expr, obj->saturate);
    }

    virtual object_t _mutate(const for_t *obj) {
        auto var = mutate(obj->var);
        auto init = mutate(obj->init);
        auto bound = mutate(obj->bound);
        auto body = mutate(obj->body);

        if (var.is_same(obj->var) && init.is_same(obj->init)
                && bound.is_same(obj->bound) && body.is_same(obj->body))
            return obj;

        return for_t::make(var, init, bound, body, obj->unroll);
    }

    virtual object_t _mutate(const func_call_t *obj) {
        auto func = mutate(obj->func);
        auto args = mutate(obj->args);

        if (func.is_same(obj->func) && ir_utils::is_same(args, obj->args))
            return obj;

        return func_call_t::make(func, args, obj->attr);
    }

    virtual object_t _mutate(const if_t *obj) {
        auto cond = mutate(obj->cond);
        auto body = mutate(obj->body);
        auto else_body = mutate(obj->else_body);

        if (cond.is_same(obj->cond) && body.is_same(obj->body)
                && else_body.is_same(obj->else_body))
            return obj;

        return if_t::make(cond, body, else_body);
    }

    virtual object_t _mutate(const iif_t *obj) {
        auto cond = mutate(obj->cond);
        auto true_expr = mutate(obj->true_expr);
        auto false_expr = mutate(obj->false_expr);

        if (cond.is_same(obj->cond) && true_expr.is_same(obj->true_expr)
                && false_expr.is_same(obj->false_expr))
            return obj;

        return iif_t::make(cond, true_expr, false_expr);
    }

    virtual object_t _mutate(const let_t *obj) {
        auto var = mutate(obj->var);
        auto value = mutate(obj->value);
        auto body = mutate(obj->body);

        if (var.is_same(obj->var) && value.is_same(obj->value)
                && body.is_same(obj->body))
            return obj;

        return let_t::make(var, value, body);
    }

    virtual object_t _mutate(const load_t *obj) {
        auto buf = mutate(obj->buf);
        auto off = mutate(obj->off);

        if (buf.is_same(obj->buf) && off.is_same(obj->off)) return obj;

        return load_t::make(obj->type, buf, off, obj->stride);
    }

    virtual object_t _mutate(const ptr_t *obj) {
        auto base = mutate(obj->base);
        auto off = mutate(obj->off);

        if (base.is_same(obj->base) && off.is_same(obj->off)) return obj;

        return ptr_t::make(base, off);
    }

    virtual object_t _mutate(const shuffle_t *obj) {
        auto vec = mutate(obj->vec);

        if (ir_utils::is_same(vec, obj->vec)) return obj;

        return shuffle_t::make(vec, obj->idx);
    }

    virtual object_t _mutate(const stmt_group_t *obj) {
        auto body = mutate(obj->body);

        if (body.is_same(obj->body)) return obj;

        return stmt_group_t::make(obj->label, body);
    }

    virtual object_t _mutate(const stmt_seq_t *obj) {
        auto head = mutate(obj->head);
        auto tail = mutate(obj->tail);

        if (head.is_same(obj->head) && tail.is_same(obj->tail)) return obj;

        return stmt_seq_t::make(head, tail);
    }

    virtual object_t _mutate(const store_t *obj) {
        auto buf = mutate(obj->buf);
        auto off = mutate(obj->off);
        auto value = mutate(obj->value);
        auto mask = mutate(obj->mask);

        if (buf.is_same(obj->buf) && off.is_same(obj->off)
                && value.is_same(obj->value) && mask.is_same(obj->mask))
            return obj;

        return store_t::make(buf, off, value, obj->stride, mask);
    }

    virtual object_t _mutate(const ternary_op_t *obj) {
        auto a = mutate(obj->a);
        auto b = mutate(obj->b);
        auto c = mutate(obj->c);

        if (a.is_same(obj->a) && b.is_same(obj->b) && c.is_same(obj->c))
            return obj;

        return ternary_op_t::make(obj->op_kind, a, b, c);
    }

    virtual object_t _mutate(const unary_op_t *obj) {
        auto a = mutate(obj->a);
        if (a.is_same(obj->a)) return obj;
        return unary_op_t::make(obj->op_kind, a);
    }

    virtual dispatch_func_type find_dispatch_func(int64_t ti) const {
        return ti < num_dispatch_funcs ? dispatch_funcs()[ti] : nullptr;
    }

private:
    static const int64_t num_dispatch_funcs
            = ir_type_id_t::end_visitable_ir_objects;
    static std::array<dispatch_func_type, num_dispatch_funcs> &
    dispatch_funcs() {
        static std::array<dispatch_func_type, num_dispatch_funcs>
                _dispatch_funcs;
        std::once_flag initialized;
        std::call_once(initialized, [&]() {
#define HANDLE_IR_OBJECT(type) \
    _dispatch_funcs[type::_dispatch_type_id()] = &call<type>;
            HANDLE_ALL_IR_OBJECTS()

#undef HANDLE_IR_OBJECT
        });
        return _dispatch_funcs;
    }

    template <typename T>
    static object_t call(ir_mutator_t *mutator, const object_impl_t *obj) {
        return mutator->_mutate((const T *)obj);
    }

    object_t dispatch(const object_impl_t *obj) {
        if (!obj) return obj;

        auto ti = obj->dispatch_type_id();
        auto f = find_dispatch_func(ti);
        ir_assert(f);
        return f(this, obj);
    }
};

class ir_context_t {
public:
    expr_t create_tmp_var(
            const type_t &type, const std::string &prefix = "tmp") {
        int &id = prefix_ids_[prefix];
        auto name = prefix + "_" + std::to_string(id);
        id++;
        return var_t::make(type, name);
    }

private:
    std::unordered_map<std::string, int> prefix_ids_;
};

class alloc_updater_t : public ir_mutator_t {
public:
    void resize(const expr_t &buf, int new_size) {
        auto ret = resizes_.insert({buf, new_size});
        ir_assert(ret.second) << buf;
        MAYBE_UNUSED(ret);
    }

    void remove(const expr_t &buf) {
        auto ret = removes_.insert(buf);
        ir_assert(ret.second) << buf;
        MAYBE_UNUSED(ret);
    }

    stmt_t update(const stmt_t &stmt) { return mutate(stmt); }

    object_t _mutate(const alloc_t *obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);

        if (try_remove(new_obj)) return new_obj;
        if (try_resize(new_obj)) return new_obj;

        return new_obj;
    }

private:
    bool try_remove(object_t &obj) {
        auto &alloc = obj.as<alloc_t>();
        auto it = removes_.find(alloc.buf);
        if (it == removes_.end()) return false;

        obj = alloc.body;
        removes_.erase(it);
        return true;
    }

    bool try_resize(object_t &obj) {
        auto &alloc = obj.as<alloc_t>();
        auto it = resizes_.find(alloc.buf);
        if (it == resizes_.end()) return false;

        obj = alloc_t::make(
                alloc.buf, it->second, alloc.kind, alloc.attr, alloc.body);
        resizes_.erase(it);
        return true;
    }

    object_set_t<expr_t> removes_;
    object_map_t<expr_t, int> resizes_;
};

template <typename T>
struct expr_cast_helper_t {
    static T call(const expr_t &e) { return to_cpp<T>(e); }

    static std::vector<T> call(const std::vector<expr_t> &exprs) {
        std::vector<T> ret;
        for (auto &e : exprs)
            ret.push_back(to_cpp<T>(e));
        return ret;
    }
};

template <>
struct expr_cast_helper_t<expr_t> {
    static expr_t call(const expr_t &e) { return e; }

    static std::vector<expr_t> call(const std::vector<expr_t> &exprs) {
        return exprs;
    }

    template <typename U,
            typename
            = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    static std::vector<expr_t> call(const std::vector<U> &vec) {
        std::vector<expr_t> ret;
        for (auto &v : vec)
            ret.push_back(to_expr(v));
        return ret;
    }
};

template <typename DstT, typename SrcT>
DstT expr_cast(const SrcT &src) {
    return expr_cast_helper_t<DstT>::call(src);
}

template <typename DstT, typename SrcT>
std::vector<DstT> expr_cast(const std::vector<SrcT> &src) {
    return expr_cast_helper_t<DstT>::call(src);
}

// Performs constant folding recursively to an IR tree.
object_t const_fold(const object_t &obj);

// Performs constant folding non-recursively to an expression.
expr_t const_fold_non_recursive(const expr_t &e);

template <typename T>
std::vector<object_t> find_objects(const object_t &root);

template <typename T>
std::vector<object_t> find_objects_unique(const object_t &root);

class alloc_manager_t {
public:
    alloc_manager_t(const stmt_t &root) {
        auto allocs = find_objects<alloc_t>(root);
        for (auto &_a : allocs) {
            auto &a = _a.as<alloc_t>();
            auto ret = buf2alloc_.insert({a.buf, _a});
            buffers_.push_back(a.buf);
            ir_assert(ret.second) << "Buffer already exists: " << a.buf;
            MAYBE_UNUSED(ret);
        }

        // Sort buffers by name.
        std::sort(buffers_.begin(), buffers_.end(),
                [](const expr_t &a, const expr_t &b) {
                    return a.as<var_t>().name < b.as<var_t>().name;
                });
    }

    const std::vector<expr_t> &buffers() const { return buffers_; }

    expr_t find_buffer(
            const std::string &name, bool allow_empty = false) const {
        for (auto &b : buffers())
            if (b.as<var_t>().name == name) return b;

        if (!allow_empty) ir_error_not_expected() << name;
        return expr_t();
    }

    std::vector<expr_t> find_buffers(alloc_kind_t kind) const {
        std::vector<expr_t> ret;
        for (auto &b : buffers())
            if (alloc_kind(b) == kind) ret.push_back(b);
        return ret;
    }

    int alloc_size(const expr_t &buf) const {
        auto *a = find_alloc(buf);
        ir_assert(a) << buf;
        return a->size;
    }

    alloc_kind_t alloc_kind(const expr_t &buf) const {
        auto *a = find_alloc(buf);
        ir_assert(a) << buf;
        return a->kind;
    }

    int total_size(alloc_kind_t kind) const {
        int ret = 0;
        for (auto &kv : buf2alloc_) {
            auto &a = kv.second.as<alloc_t>();
            if (a.kind == kind) ret += a.size;
        }
        return ret;
    }

private:
    const alloc_t *find_alloc(const expr_t &buf) const {
        auto it = buf2alloc_.find(buf);
        if (it == buf2alloc_.end()) return nullptr;
        return it->second.as_ptr<alloc_t>();
    }

    object_map_t<expr_t, stmt_t> buf2alloc_;
    std::vector<expr_t> buffers_;
    object_map_t<expr_t, stmt_t> alloc_updates_;
};

// IR utility functions.
expr_t abs(const expr_t &e);

expr_t cast(const expr_t &e, const type_t &type, bool saturate = false);

bool is_zero(const expr_t &e);

bool is_one(const expr_t &e);

bool is_minus_one(const expr_t &e);

bool is_const_broadcast(const expr_t &e);

bool is_const_broadcast(const expr_t &e, const expr_t &value);

bool all_of(const expr_t &e, const expr_t &value);

expr_t make_buffer(const std::string &name);

// Utility functions for nary_op_t.
expr_t nary_op_back_transform(const expr_t &e);
expr_t nary_op_canonicalize(const expr_t &_e);
expr_t make_nary_op(op_kind_t op_kind, const std::vector<expr_t> &args);
std::vector<expr_t> cvt_expr_to_nary_op_args(const expr_t &e);

// Substitutes all occurrences of `from` to `to` in `root.
object_t substitute(const object_t &root, const object_t &from,
        const object_t &to,
        int max_substitutions = std::numeric_limits<int>::max());

// Returns leaf statements of `root`. Uses inorder traversal.
std::vector<stmt_t> flatten_statements(const stmt_t &root);

template <typename T, bool find_unique = false, bool save_objects = true>
class object_finder_t : public ir_visitor_t {
public:
    void _visit(const T *obj) override {
        ir_visitor_t::_visit(obj);
        occurrences++;
        if (!save_objects) return;
        if (find_unique) {
            found_unique.insert(obj);
        } else {
            found.push_back(obj);
        }
    }

    std::vector<object_t> found;
    object_set_t<object_t> found_unique;
    int occurrences = 0;
};

// Returns all IR objects of type `T` found in `root`.
template <typename T>
std::vector<object_t> find_objects(const object_t &root) {
    object_finder_t<T, /*find_unique=*/false> finder;
    finder.visit(root);
    return finder.found;
}

template <typename T>
int count_objects(const object_t &root) {
    object_finder_t<T, /*find_unique=*/false, /*save_objects=*/false> finder;
    finder.visit(root);
    return finder.occurrences;
}

// Returns unique IR objects of type `T` found in `root`.
template <typename T>
object_set_t<object_t> find_unique_objects(const object_t &root) {
    object_finder_t<T, /*find_unique=*/true> finder;
    finder.visit(root);
    return finder.found_unique;
}

// Returns number of occurrences of `obj` in `root` (based on identity
// comparison).
int count_object(const object_t &root, const object_t &obj);

// Returns number of occurrences of `obj` in vector of root objects (based on
// identity comparison).
template <typename T>
int count_object(const std::vector<T> &roots, const object_t &obj) {
    int ret = 0;
    for (auto &root : roots)
        ret += count_object(root, obj);
    return ret;
}

// Checks if `root` contains `obj`.
bool contains_object(const object_t &root, const object_t &obj);

// Returns all statement groups matching the label.
std::vector<stmt_t> find_stmt_groups(
        const object_t &root, const stmt_label_t &label);

// Returns a statement group matching the label. `root` must have exactly one
// occurrence.
stmt_t find_stmt_group(const object_t &root, const stmt_label_t &label);

class scope_visitor_t : public ir_visitor_t {
public:
    bool is_expr_defined(const expr_t &e) const {
        auto vars = find_unique_objects<var_t>(e);
        for (auto &v : vars) {
            if (def_vars_.count(v) == 0) return false;
        }
        return true;
    }

#define CASE(type, var_field, is_pre) \
    if (obj->type_id() == type::_type_id()) { \
        visit_scope( \
                (const type *)obj, ((const type *)obj)->var_field, is_pre); \
        return; \
    }

    void pre_visit(const object_impl_t *obj) override {
        CASE(alloc_t, buf, true);
        CASE(let_t, var, true);
        CASE(for_t, var, true);
    }

    void post_visit(const object_impl_t *obj) override {
        CASE(alloc_t, buf, false);
        CASE(let_t, var, false);
        CASE(for_t, var, false);
    }

#undef CASE

private:
    template <typename T>
    void visit_scope(const T *obj, const expr_t &var, bool is_pre_visit) {
        if (is_pre_visit) {
            def_vars_.insert(var);
            return;
        }
        def_vars_.erase(var);
    }

    object_set_t<expr_t> def_vars_;
};

class ir_path_t {
public:
    void push(const object_impl_t *obj) { path_.push_back(obj); }

    void pop() { path_.pop_back(); }

    const object_impl_t *back() const {
        ir_assert(!is_empty());
        return path_.back();
    }

    bool is_empty() const { return path_.empty(); }

    void merge(const ir_path_t &other) {
        size_t idx;
        size_t min_size = std::min(path_.size(), other.path_.size());
        for (idx = 0; idx < min_size; idx++) {
            if (path_[idx] != other.path_[idx]) break;
        }
        path_.resize(idx);
    }

private:
    std::vector<const object_impl_t *> path_;
};

// Only for statements that create scope.
stmt_t get_stmt_body(const stmt_t &stmt);

stmt_t replace_stmt_body(const stmt_t &stmt, const stmt_t &new_body);

// Describes the linear transformation F(x) for variable x: F(x) = (a * x + b),
// where a and b are integer constants.
struct linear_transform_t {
    expr_t x;
    int a;
    int b;

    bool is_identity() const { return a == 1 && b == 0; }
};

// Relation: (lhs op rhs), where:
// - lhs is a variable
// - rhs is an integer constant
// - op is a comparison operation
class relation_t {
public:
    relation_t(const expr_t &expr) : expr_(normalize(expr)) {}

    const expr_t &expr() const { return expr_; }

    const expr_t &var() const { return expr_.as<binary_op_t>().a; }

    const expr_t &rhs() const { return expr_.as<binary_op_t>().b; }

    op_kind_t op_kind() const { return expr_.as<binary_op_t>().op_kind; }

    bool implies(const relation_t &other) const;

    // Applies linear transformation to left and right hand sides of the relation.
    relation_t transform(const linear_transform_t &t, const expr_t &new_var);

    std::string str() const {
        std::ostringstream oss;
        oss << expr_;
        return oss.str();
    }

    static bool is_relation_constraint(const expr_t &e) {
        auto *binary_op = e.as_ptr<binary_op_t>();
        if (!binary_op) return false;
        if (!is_var(binary_op->a)) return false;
        if (!is_const(binary_op->b)) return false;
        if (!is_cmp_op(binary_op->op_kind)) return false;
        return true;
    }

private:
    static expr_t normalize(const expr_t &e);

    expr_t expr_;
};

inline std::ostream &operator<<(std::ostream &out, const relation_t &rel) {
    out << rel.str();
    return out;
}

// Equality for modulus: (var % mod) == 0, where:
// - var is a variable
// - mod is an integer constant
class modulus_info_t {
public:
    modulus_info_t(const expr_t &expr) : expr_(expr) {}

    const expr_t &expr() const { return expr_; }

    const expr_t &var() const {
        auto &mod_expr = expr_.as<binary_op_t>().a;
        return mod_expr.as<binary_op_t>().a;
    }

    const expr_t &mod() const {
        auto &mod_expr = expr_.as<binary_op_t>().a;
        return mod_expr.as<binary_op_t>().b;
    }

    bool implies(const modulus_info_t &other) const {
        ir_assert(var().is_same(other.var()));

        int64_t this_mod = to_cpp<int64_t>(mod());
        int64_t other_mod = to_cpp<int64_t>(other.mod());

        return this_mod % other_mod == 0;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << expr_;
        return oss.str();
    }

    // Try to match (var % mod) == 0.
    static bool is_modulus_constraint(const expr_t &e);

private:
    expr_t expr_;
};

inline std::ostream &operator<<(std::ostream &out, const modulus_info_t &mod) {
    out << mod.str();
    return out;
}

// TODO: Add integers check (only integers can be constrained).
class constraint_set_t {
public:
    void add_constraint(const expr_t &e);

    bool can_prove(const expr_t &e, bool try_simplify = true) const {
        auto ret = can_prove_impl(e, /*do_simplify=*/false);
        if (ret || !try_simplify) return ret;

        return can_prove_impl(e, /*do_simplify=*/true);
    }

    bool is_single_value(const expr_t &e, expr_t &value) const;

    int max_proven_gcd(const expr_t &var) const;

private:
    bool can_prove_modulus(const expr_t &e) const {
        modulus_info_t unknown(e);
        auto it = modulus_infos_.find(unknown.var());
        if (it == modulus_infos_.end()) return false;

        for (auto &known : it->second) {
            if (known.implies(unknown)) return true;
        }

        return false;
    }

    bool can_prove_relation(const expr_t &e) const {
        relation_t unknown(e);
        auto it = relations_.find(unknown.var());
        if (it == relations_.end()) return false;

        for (auto &known : it->second) {
            if (known.implies(unknown)) return true;
        }

        return false;
    }

    bool can_prove_impl(const expr_t &_e, bool do_simplify) const;

    object_map_t<expr_t, std::vector<relation_t>> relations_;
    object_map_t<expr_t, std::vector<modulus_info_t>> modulus_infos_;
};

// Simplifies expression or statement. An optional constraint set is used to
// pass known equalities and inequalities which may be used for simplification.
object_t simplify(const object_t &obj, const constraint_set_t &cset = {});

// Searches for expression patterns to reduce them to the equivalent ternary
// operations.
expr_t simplify_rewrite_with_ternary(const expr_t &e, bool recursive = true);

// Moves constants to the right hand side of an expression.
// Example: (c0 + x) op c1 -> x op (c1 - c0)
expr_t simplify_cmp_move_const_to_rhs(const expr_t &e);

// Reduces left and right hand sides of an expression.
// Example: A * x < A * B -> x < B (if A > 0).
expr_t simplify_cmp_reduce_lhs_rhs(const expr_t &e);

// Propagates shuffle down the expression tree for more effective vectorization.
expr_t simplify_propagate_shuffle(const expr_t &e);

// Pre-defined functions.
namespace funcs {

inline func_t barrier_func() {
    static auto f = builtin_t::make("barrier");
    return f;
}

inline stmt_t barrier() {
    return barrier_func().call();
}

inline func_t slm_fence_func() {
    static auto f = builtin_t::make("slm_fence");
    return f;
}

inline stmt_t slm_fence() {
    return slm_fence_func().call();
}

inline func_t signal_func() {
    static auto f = builtin_t::make("signal");
    return f;
}

inline stmt_t signal() {
    return signal_func().call();
}

inline func_t barrier_wait_func() {
    static auto f = builtin_t::make("barrier_wait");
    return f;
}

inline stmt_t barrier_wait() {
    return barrier_wait_func().call();
}

} // namespace funcs

// Helper functionality to extract ND indices packed into 1D index.
// Example:
//     i = [0; Bi, 2 * Bi, ... (I - 1) * Bi]
//     i_info.dim = I; i_info.block = Bi
//     j = [0; Bj, 2 * Bj, ... (J - 1) * Bj]
//     j_info.dim = J; j_info.block = Bj
// 1D index: ij_idx
// 2D indices: [i; j]
// Unpacking:
//     i = (ij_idx % I) * Bi
//     j = (ij_idx / I) * Bj
struct unpack_dim_info_t {
    const expr_t &var;
    int dim;
    int block;
};

inline void cvt_args_to_unpack_dim_info(std::vector<unpack_dim_info_t> &) {}

template <typename... ArgsT>
void cvt_args_to_unpack_dim_info(std::vector<unpack_dim_info_t> &infos,
        const expr_t &var, int dim, int block, const ArgsT &... args) {
    infos.push_back(unpack_dim_info_t {var, dim, block});
    cvt_args_to_unpack_dim_info(infos, args...);
}

void unpack(std::vector<stmt_t> &init_stmts, constraint_set_t &cset,
        const expr_t &_e, const std::vector<unpack_dim_info_t> &infos);

template <typename... ArgsT>
void unpack(std::vector<stmt_t> &init_stmts, constraint_set_t &cset,
        const expr_t &e, const ArgsT &... args) {
    std::vector<unpack_dim_info_t> infos;
    cvt_args_to_unpack_dim_info(infos, args...);
    unpack(init_stmts, cset, e, infos);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
