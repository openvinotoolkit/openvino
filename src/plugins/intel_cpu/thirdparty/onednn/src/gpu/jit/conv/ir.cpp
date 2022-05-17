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

#include <sstream>

#include "common/math_utils.hpp"
#include "gpu/jit/conv/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ir_utils;

namespace {

// Helper class to print IR objects.
class ir_printer_t : public ir_visitor_t {
public:
    ir_printer_t(std::ostream &out) : out_(out) {}

    void visit(const object_t &obj) override {
        if (is_supported(obj)) {
            ir_visitor_t::visit(obj);
            return;
        }
        // Only expressions/functions are expected here.
        out_ << obj.str();
    }

    void _visit(const alloc_t *obj) override {
        print_indent();
        out_ << "alloc " << obj->buf.as<var_t>().name << "[" << obj->size
             << "]\n";
        visit(obj->body);
    }

    void _visit(const binary_op_t *obj) override {
        if (utils::one_of(obj->op_kind, op_kind_t::_min, op_kind_t::_max)) {
            out_ << to_string(obj->op_kind) << "(" << obj->a << ", " << obj->b
                 << ")";
            return;
        }
        out_ << "(";
        visit(obj->a);
        out_ << " " << to_string(obj->op_kind) << " ";
        visit(obj->b);
        out_ << ")";
    }

    void _visit(const bool_imm_t *obj) override {
        out_ << (obj->value ? "true" : "false");
    }

    void _visit(const cast_t *obj) override {
        out_ << obj->type;
        if (obj->saturate) out_ << ".sat";
        out_ << "(" << obj->expr << ")";
    }

    void _visit(const float_imm_t *obj) override { out_ << obj->value; }

    void _visit(const for_t *obj) override {
        print_indent();
        out_ << "for (" << obj->var << " = " << obj->init << "; " << obj->var
             << " < " << obj->bound << "; " << obj->var << "++) ";
        if (obj->unroll != 1) out_ << "[unroll: " << obj->unroll << "] ";
        out_ << "{\n";
        add_indent();
        visit(obj->body);
        remove_indent();
        print_indent();
        out_ << "}\n";
    }

    void _visit(const func_call_t *obj) override {
        print_indent();
        out_ << obj->func << "(" << make_seq_print_helper(obj->args) << ")";
        if (!obj->attr.is_empty()) out_ << " " << obj->attr;
        out_ << "\n";
    }

    void _visit(const func_impl_t *obj) override { out_ << obj->str(); }

    void _visit(const if_t *obj) override {
        print_indent();
        out_ << "if (" << strip_parens(obj->cond.str()) << ") {\n";
        add_indent();
        visit(obj->body);
        remove_indent();
        print_indent();
        if (obj->else_body.is_empty()) {
            out_ << "}\n";
            return;
        }
        out_ << "} else {\n";
        add_indent();
        visit(obj->else_body);
        remove_indent();
        print_indent();
        out_ << "}\n";
    }

    void _visit(const iif_t *obj) override {
        out_ << "(" << obj->cond << " ? " << obj->true_expr << " : "
             << obj->false_expr << ")";
    }

    void _visit(const int_imm_t *obj) override {
        out_ << std::to_string(obj->value);
    }

    void _visit(const let_t *obj) override {
        print_indent();
        out_ << obj->var << "." << obj->var.type() << " = " << obj->value
             << "\n";
        visit(obj->body);
    }

    void _visit(const load_t *obj) override {
        out_ << obj->buf;
        if (obj->has_default_stride()) {
            out_ << "." << obj->type << "(" << obj->off / obj->type.size()
                 << ")";
        } else {
            out_ << "[" << obj->off << "]." << obj->type;
            out_ << "<" << obj->stride << ">";
        }
    }

    void _visit(const ptr_t *obj) override {
        out_ << obj->base << "[" << obj->off << "]";
    }

    void _visit(const shuffle_t *obj) override {
        if (obj->is_broadcast()) {
            out_ << "bcast" << obj->elems() << "(" << obj->vec[0] << ")";
            return;
        }
        std::vector<expr_t> vec_all;
        for (auto &v : obj->vec) {
            for (int i = 0; i < v.type().elems(); i++)
                vec_all.push_back(v);
        }
        int elems = obj->type.elems();
        out_ << "(";
        for (int i = 0; i < elems; i++) {
            int idx = obj->idx[i];
            auto &v = vec_all[idx];
            int v_elems = v.type().elems();
            out_ << v;
            if (v_elems != 1) out_ << "[" << idx << "]";
            if (i != elems - 1) out_ << ", ";
        }
        out_ << ")";
    }

    void _visit(const stmt_group_t *obj) override {
        print_indent();
        out_ << obj->label << " {\n";
        add_indent();
        visit(obj->body);
        remove_indent();
        print_indent();
        out_ << "}\n";
        return;
    }

    void _visit(const stmt_seq_t *obj) override {
        visit(obj->head);
        visit(obj->tail);
    }

    void _visit(const store_t *obj) override {
        print_indent();
        out_ << load_t::make(
                obj->value.type(), obj->buf, obj->off, obj->stride);
        out_ << " = " << obj->value;
        if (!obj->mask.is_empty()) out_ << " [masked]";
        out_ << "\n";
    }

    void _visit(const ternary_op_t *obj) override {
        out_ << to_string(obj->op_kind) << "(" << obj->a << ", " << obj->b
             << ", " << obj->c << ")";
        return;
    }

    void _visit(const unary_op_t *obj) override {
        out_ << to_string(obj->op_kind);
        visit(obj->a);
    }

    void _visit(const var_t *obj) override { out_ << obj->name; }

private:
    static std::string strip_parens(const std::string &s) {
        if (s.size() < 2 || s[0] != '(' || s[s.size() - 1] != ')') return s;
        auto ret = s;
        ret.resize(s.size() - 1);
        return ret.substr(1);
    }

    void print_indent() {
        for (int i = 0; i < indent_; i++)
            out_ << prefix_;
    }

    void add_indent() { indent_++; }
    void remove_indent() { indent_--; }

    std::ostream &out_;
    int indent_ = 0;

    std::string prefix_ = "  ";
};

class substitute_mutator_t : public ir_mutator_t {
public:
    substitute_mutator_t(const object_t &from, const object_t &to)
        : from_(from), to_(to) {}

    int substitutions() const { return substitutions_; }

    dispatch_func_type find_dispatch_func(int64_t) const override {
        return mutate_object;
    }

    static object_t mutate_object(
            ir_mutator_t *mutator, const object_impl_t *obj) {
        auto *this_mutator = (substitute_mutator_t *)mutator;

        if (this_mutator->from_.is_same(obj)) {
            this_mutator->substitutions_++;
            return this_mutator->to_;
        }

        auto ti = obj->dispatch_type_id();
        auto f = mutator->ir_mutator_t::find_dispatch_func(ti);
        return f(mutator, obj);
    }

private:
    object_t from_;
    object_t to_;

    int substitutions_ = 0;
};

class stmt_flattener_t : public ir_visitor_t {
public:
#define HANDLE_IR_OBJECT(type) \
    void _visit(const type *obj) { \
        size_t old_size = stmts.size(); \
        ir_visitor_t::_visit(obj); \
        if (stmts.size() > old_size) return; \
        if (obj->is_stmt()) stmts.push_back(obj); \
    }

    HANDLE_ALL_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    std::vector<stmt_t> stmts;
};

} // namespace

std::string object_impl_t::str() const {
    std::ostringstream oss;
    ir_printer_t printer(oss);
    ir_assert(printer.is_supported(this));
    printer.visit(this);
    return oss.str();
}

object_t substitute(const object_t &root, const object_t &from,
        const object_t &to, int max_substitutions) {
    if (to.is_same(from)) return root;
    substitute_mutator_t sm(from, to);
    auto ret = sm.mutate(root);
    ir_assert(sm.substitutions() <= max_substitutions)
            << "Unexpected number of substitutions.";
    MAYBE_UNUSED(&substitute_mutator_t::substitutions);
    MAYBE_UNUSED(max_substitutions);
    return ret;
}

std::vector<stmt_t> flatten_statements(const stmt_t &root) {
    stmt_flattener_t f;
    f.visit(root);
    return f.stmts;
}

expr_t abs(const expr_t &e) {
    ir_assert(is_const(e)) << e;
    if (to_cpp<bool>(e >= 0)) return e;
    return -e;
}

expr_t cast(const expr_t &e, const type_t &type, bool saturate) {
    if (e.type() == type) return e;
    return const_fold(cast_t::make(type, e, saturate));
}

bool is_zero(const expr_t &e) {
    if (!e.type().is_scalar()) return false;
    return e.is_equal(to_expr(0, e.type()));
}

bool is_one(const expr_t &e) {
    if (!e.type().is_scalar()) return false;
    return e.is_equal(to_expr(1, e.type()));
}

bool is_minus_one(const expr_t &e) {
    if (!e.type().is_scalar()) return false;
    return e.is_equal(to_expr(-1, e.type()));
}

bool is_const_broadcast(const expr_t &e) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return false;
    if (!shuffle->is_broadcast()) return false;
    return is_const(shuffle->vec[0]);
}

bool is_const_broadcast(const expr_t &e, const expr_t &value) {
    if (!is_const_broadcast(e)) return false;
    return e.as<shuffle_t>().vec[0].is_equal(value);
}

bool all_of(const expr_t &e, const expr_t &value) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (!shuffle) return e.is_equal(value);
    for (auto &i : shuffle->idx) {
        if (!shuffle->vec[i].is_equal(value)) return false;
    }
    return true;
}

expr_t make_buffer(const std::string &name) {
    return var_t::make(type_t::byte_ptr(), name);
}

// Returns number of occurrences of `obj` in `root` (based on identity equality).
int count_object(const object_t &root, const object_t &obj) {
    ir_assert(!obj.is_empty());

    std::vector<object_t> found;
    do {
#define HANDLE_IR_OBJECT(type) \
    if (obj.dispatch_type_id() == type::_dispatch_type_id()) { \
        found = find_objects<type>(root); \
        break; \
    }

        HANDLE_ALL_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

        ir_error_not_expected() << obj;
    } while (false);

    int ret = 0;
    for (auto &f : found)
        if (f.is_equal(obj)) ret++;
    return ret;
}

bool contains_object(const object_t &root, const object_t &obj) {
    ir_assert(is_var(obj)) << obj;
    return count_object(root, obj) > 0;
}

std::vector<stmt_t> find_stmt_groups(
        const object_t &root, const stmt_label_t &label) {
    auto groups = find_objects<stmt_group_t>(root);
    std::vector<stmt_t> ret;
    for (auto &g : groups) {
        if (g.as<stmt_group_t>().label == label) ret.push_back(g);
    }
    return ret;
}

stmt_t find_stmt_group(const object_t &root, const stmt_label_t &label) {
    auto groups = find_stmt_groups(root, label);
    ir_assert(groups.size() == 1);
    return groups[0];
}

stmt_t get_stmt_body(const stmt_t &stmt) {
    auto *alloc = stmt.as_ptr<alloc_t>();
    if (alloc) return alloc->body;

    auto *_for = stmt.as_ptr<for_t>();
    if (_for) return _for->body;

    auto *let = stmt.as_ptr<let_t>();
    if (let) return let->body;

    auto *group = stmt.as_ptr<stmt_group_t>();
    if (group) return group->body;

    return stmt;
}

stmt_t replace_stmt_body(const stmt_t &stmt, const stmt_t &new_body) {
    auto *alloc = stmt.as_ptr<alloc_t>();
    if (alloc) {
        return alloc_t::make(
                alloc->buf, alloc->size, alloc->kind, alloc->attr, new_body);
    }

    auto *_for = stmt.as_ptr<for_t>();
    if (_for) {
        return for_t::make(
                _for->var, _for->init, _for->bound, new_body, _for->unroll);
    }

    auto *let = stmt.as_ptr<let_t>();
    if (let) { return let_t::make(let->var, let->value, new_body); }

    auto *group = stmt.as_ptr<stmt_group_t>();
    if (group) { return stmt_group_t::make(group->label, new_body); }

    return new_body;
}

bool relation_t::implies(const relation_t &other) const {
    ir_assert(var().is_same(other.var()));

    if (op_kind() != other.op_kind()) return false;

    auto A = to_cpp<int64_t>(rhs());
    auto B = to_cpp<int64_t>(other.rhs());

    switch (op_kind()) {
        // (x > A) && (A >= B) => (x > B)
        // (x >= A) && (A >= B) => (x >= B)
        case op_kind_t::_gt:
        case op_kind_t::_ge: return A >= B;
        // (x < A) && (A <= B) => (x < B)
        // (x <= A) && (A <= B) => (x <= B)
        case op_kind_t::_lt:
        case op_kind_t::_le: return A <= B;
        default: ir_error_not_expected() << "Not implemented: " << expr_;
    }
    return false;
}

relation_t relation_t::transform(
        const linear_transform_t &t, const expr_t &new_var) {
    ir_assert(t.a == 1) << "Not implemented.";
    return relation_t(binary_op_t::make(op_kind(), new_var, rhs() + t.b));
}

expr_t relation_t::normalize(const expr_t &e) {
    ir_assert(is_relation_constraint(e)) << e;
    auto &op = e.as<binary_op_t>();

    auto op_kind = op.op_kind;
    auto a = op.a;
    auto b = op.b;

    switch (op_kind) {
        case op_kind_t::_lt:
            op_kind = op_kind_t::_le;
            b -= 1;
            break;
        case op_kind_t::_gt:
            op_kind = op_kind_t::_ge;
            b += 1;
            break;
        default: return e;
    }
    return binary_op_t::make(op_kind, a, b);
}

bool modulus_info_t::is_modulus_constraint(const expr_t &e) {
    auto *binary_op = e.as_ptr<binary_op_t>();
    if (!binary_op) return false;
    if (!is_zero(binary_op->b)) return false;
    if (binary_op->op_kind != op_kind_t::_eq) return false;

    auto *mod_op = binary_op->a.as_ptr<binary_op_t>();
    if (!mod_op) return false;
    if (mod_op->op_kind != op_kind_t::_mod) return false;
    if (!is_var(mod_op->a)) return false;
    if (!is_const(mod_op->b)) return false;

    return true;
}

bool is_linear_var_transform(const expr_t &e, linear_transform_t &t) {
    if (is_var(e)) {
        t.x = e;
        t.a = 1;
        t.b = 0;
        return true;
    }

    auto *binary_op = e.as_ptr<binary_op_t>();
    if (!binary_op) return false;

    auto vars = find_objects<var_t>(e);
    if (vars.size() != 1) return false;

    auto &var = vars[0];

    // TODO: Extend to match multiplication: (a * var).
    if (!utils::one_of(binary_op->op_kind, op_kind_t::_add, op_kind_t::_sub))
        return false;

    auto &a = binary_op->a;
    auto &b = binary_op->b;

    bool is_sub = (binary_op->op_kind == op_kind_t::_sub);

    // var op b -> (t.a = 1, t.b = +/-b)
    if (a.is_same(var) && is_const(b)) {
        t.x = var;
        t.a = 1;
        t.b = (is_sub ? -1 : 1) * to_cpp<int>(b);
        return true;
    }

    // a op var -> (t.a = +/-1, t.b = a)
    if (is_const(a) && b.is_same(var)) {
        t.x = var;
        t.a = (is_sub ? -1 : 1);
        t.b = to_cpp<int>(a);
        return true;
    }

    return false;
}

void constraint_set_t::add_constraint(const expr_t &e) {
    auto *shuffle = e.as_ptr<shuffle_t>();
    if (shuffle) {
        if (shuffle->is_broadcast()) add_constraint(shuffle->vec[0]);
        return;
    }

    if (modulus_info_t::is_modulus_constraint(e)) {
        modulus_info_t mi(e);
        modulus_infos_[mi.var()].push_back(mi);
        return;
    }

    if (relation_t::is_relation_constraint(e)) {
        relation_t rel(e);
        relations_[rel.var()].push_back(rel);
        return;
    }

    // Propagate constraints from y for (x == y) equalities.
    auto *binary_op = e.as_ptr<binary_op_t>();
    if (binary_op && binary_op->op_kind == op_kind_t::_eq) {
        auto &a = binary_op->a;
        auto &b = binary_op->b;
        linear_transform_t t;
        if (is_var(a) && is_linear_var_transform(b, t)) {
            // Relations.
            auto r_it = relations_.find(t.x);
            if (r_it != relations_.end()) {
                for (auto &c : r_it->second) {
                    add_constraint(c.transform(t, a).expr());
                }
            }
            // Modulus.
            if (t.is_identity()) {
                auto m_it = modulus_infos_.find(t.x);
                if (m_it != modulus_infos_.end()) {
                    for (auto &c : m_it->second) {
                        add_constraint(substitute(c.expr(), b, a));
                    }
                }
            }
            return;
        }
    }
}

bool constraint_set_t::is_single_value(const expr_t &e, expr_t &value) const {
    ir_assert(is_var(e)) << e;
    auto it = relations_.find(e);
    if (it == relations_.end()) return false;

    expr_t lo;
    expr_t hi;
    for (auto &rel : it->second) {
        ir_assert(is_const(rel.rhs())) << rel;
        bool do_break = false;
        switch (rel.op_kind()) {
            case op_kind_t::_eq:
                lo = hi = rel.rhs();
                do_break = true;
                break;
            case op_kind_t::_ge:
            case op_kind_t::_gt: {
                auto cur_lo = (rel.op_kind() == op_kind_t::_ge ? rel.rhs()
                                                               : rel.rhs() + 1);
                if (lo.is_empty() || to_cpp<bool>(cur_lo > lo)) { lo = cur_lo; }
                break;
            }
            case op_kind_t::_le:
            case op_kind_t::_lt: {
                auto cur_hi = (rel.op_kind() == op_kind_t::_le ? rel.rhs()
                                                               : rel.rhs() - 1);
                if (hi.is_empty() || to_cpp<bool>(cur_hi < hi)) { hi = cur_hi; }
                break;
            }
            default: ir_error_not_expected() << rel;
        }
        if (do_break) break;
    }
    bool ret = !lo.is_empty() && lo.is_equal(hi);
    if (ret) value = lo;
    return ret;
}

bool constraint_set_t::can_prove_impl(
        const expr_t &_e, bool do_simplify) const {
    auto e = _e;
    if (is_const(e)) {
        ir_assert(e.type() == type_t::_bool()) << e;
        return to_cpp<bool>(e);
    }

    if (do_simplify) {
        // These passes for comparison help to prove more inequalities.
        e = simplify_cmp_move_const_to_rhs(e);
        e = simplify_cmp_reduce_lhs_rhs(e);
        e = simplify(e);
        if (is_const(e)) {
            ir_assert(e.type() == type_t::_bool()) << e;
            return to_cpp<bool>(e);
        }
    }

    if (modulus_info_t::is_modulus_constraint(e)) return can_prove_modulus(e);
    if (relation_t::is_relation_constraint(e)) return can_prove_relation(e);

    // Can't prove.
    return false;
}

int constraint_set_t::max_proven_gcd(const expr_t &var) const {
    auto it = modulus_infos_.find(var);
    if (it == modulus_infos_.end()) return 1;
    int ret = 1;
    for (auto &c : it->second) {
        ret = math::lcm(ret, to_cpp<int>(c.mod()));
    }
    return ret;
}

void unpack(std::vector<stmt_t> &init_stmts, constraint_set_t &cset,
        const expr_t &_e, const std::vector<unpack_dim_info_t> &infos) {
    int elems = 1;
    for (auto &info : infos)
        elems *= info.dim;
    ir_assert(elems >= 1);

    expr_t e = _e;
    int rem_elems = elems;
    for (auto &info : infos) {
        auto &var = info.var;
        int dim = info.dim;
        int block = info.block;
        expr_t value;
        if (dim == 1) {
            value = expr_t(0);
            cset.add_constraint(var == 0);
        } else {
            value = block * (rem_elems > dim ? e % dim : e);
            e = e / dim;
        }
        init_stmts.emplace_back(let_t::make(var, value));
        if (dim > 1 && block > 1) cset.add_constraint(var % block == 0);
        rem_elems /= dim;
    }
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
