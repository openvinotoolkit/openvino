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

#ifndef GPU_JIT_CONV_GEMM_SCHEDULE_HPP
#define GPU_JIT_CONV_GEMM_SCHEDULE_HPP

#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <initializer_list>

#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Used to describe semantics of a dimension in the GEMM context.
// GEMM operation is defined as C = A x B
// GEMM dimension kinds:
// - B:  shared by all tensors A, B, C (batch dimension)
// - M:  shared only by A and C
// - N:  shared only by B and C
// - K:  shared only by A and B (reduction dimension)
enum class bmnk_kind_t { undef = -1, b = 0, m = 1, n = 2, k = 3 };

enum class abc_kind_t { undef, a, b, c };

class bmnk_mapper_t {
public:
    bmnk_mapper_t() = default;

    bmnk_mapper_t(const object_map_t<expr_t, bmnk_kind_t> &bmnk_kinds)
        : bmnk_kinds_(bmnk_kinds) {}

    bmnk_kind_t bmnk_kind(const expr_t &var) const {
        auto it = bmnk_kinds_.find(var);
        if (it == bmnk_kinds_.end()) return bmnk_kind_t::undef;
        return it->second;
    }

    bmnk_kind_t bmnk_kind(abc_kind_t abc_kind, int dim_idx) const {
        return bmnk_kind(var(abc_kind, dim_idx));
    }

    int ndims(abc_kind_t abc_kind) const {
        return int(get_vars(abc_kind).size());
    }

    void set_a_vars(const std::vector<expr_t> &vars) { a_vars_ = vars; }
    void set_b_vars(const std::vector<expr_t> &vars) { b_vars_ = vars; }
    void set_c_vars(const std::vector<expr_t> &vars) { c_vars_ = vars; }

    void set_bmnk_kind(const expr_t &var, bmnk_kind_t bmnk_kind) {
        auto ret = bmnk_kinds_.insert({var, bmnk_kind});
        ir_assert(ret.second) << "Can't set variable twice: " << var;
    }

    const expr_t &var(abc_kind_t abc_kind, int dim_idx) const {
        return get_vars(abc_kind)[dim_idx];
    }

    int dim_idx(abc_kind_t abc_kind, const expr_t &var) const {
        auto &vars = get_vars(abc_kind);
        for (int i = 0; i < int(vars.size()); i++) {
            if (vars[i].is_same(var)) return i;
        }
        return -1;
    }

    layout_t map_to_bmnk(abc_kind_t abc_kind,
            const std::vector<bmnk_kind_t> &bmnk_kinds,
            const view_t &view) const;

    layout_t map_to_bmnk(abc_kind_t abc_kind,
            const std::vector<bmnk_kind_t> &bmnk_kinds,
            const layout_t &layout) const;

private:
    const std::vector<expr_t> &get_vars(abc_kind_t abc_kind) const {
        switch (abc_kind) {
            case abc_kind_t::a: return a_vars_;
            case abc_kind_t::b: return b_vars_;
            case abc_kind_t::c: return c_vars_;
            default: ir_error_not_expected() << "Unknown ABC kind.";
        }
        return a_vars_;
    }

    std::vector<expr_t> &get_vars(abc_kind_t abc_kind) {
        auto &vars
                = const_cast<const bmnk_mapper_t *>(this)->get_vars(abc_kind);
        return const_cast<std::vector<expr_t> &>(vars);
    }

    std::vector<expr_t> a_vars_;
    std::vector<expr_t> b_vars_;
    std::vector<expr_t> c_vars_;
    object_map_t<expr_t, bmnk_kind_t> bmnk_kinds_;
};

class bmnk_block_mapper_t {
public:
    bmnk_block_mapper_t(const bmnk_mapper_t &bmnk_mapper)
        : bmnk_mapper_(bmnk_mapper) {}

    void push_blocks(abc_kind_t abc_kind, const std::vector<block_t> &blocks) {
        for (auto &b : blocks)
            push_block(abc_kind, b);
    }

    void push_block(abc_kind_t abc_kind, const block_t &b);

    layout_t map_from_bmnk(abc_kind_t abc_kind,
            const std::vector<bmnk_kind_t> &bmnk_kinds,
            const layout_t &bmnk_layout) const;

private:
    static void pop_size_1_blocks(std::vector<block_t> &blocks) {
        while (!blocks.empty() && blocks.front().block == 1) {
            blocks.erase(blocks.begin());
        }
    }

    std::vector<block_t> create_prb_blocks(abc_kind_t abc_kind,
            const std::vector<std::pair<abc_kind_t, block_t>> &mn_blocks)
            const {
        std::vector<block_t> ret;
        ret.reserve(mn_blocks.size());
        for (auto &p : mn_blocks) {
            auto b = p.second;
            const auto &var = bmnk_mapper_.var(p.first, b.dim_idx);
            b.dim_idx = bmnk_mapper_.dim_idx(abc_kind, var);
            ret.push_back(b);
        }
        return ret;
    }

    bool pop_block(std::vector<block_t> &bmnk_blocks,
            std::vector<block_t> &prb_blocks, const block_t &bmnk_block) const;

    bmnk_mapper_t bmnk_mapper_;

    // Ordered from innermost to outermost.
    std::vector<std::pair<abc_kind_t, block_t>> m_blocks_;
    std::vector<std::pair<abc_kind_t, block_t>> n_blocks_;
    std::vector<std::pair<abc_kind_t, block_t>> k_blocks_;
};

enum class loop_kind_t : int {
    undef,
    kernel_grid, // Loop is bound to the kernel grid.
    serial, // Loop is inside a thread (may be unrolled or just a regular loop).
    tg_grid, // Loop is bound to the thread group grid.
    tensorized, // Such loops are fully unrolled/vectorized and converted to blocked multiplication.
};

static std::string to_string(loop_kind_t kind) {
    switch (kind) {
        case loop_kind_t::undef: return "undef";
        case loop_kind_t::kernel_grid: return "kernel_grid";
        case loop_kind_t::serial: return "serial";
        case loop_kind_t::tg_grid: return "tg_grid";
        case loop_kind_t::tensorized: return "tensorized";
        default: ir_error_not_expected();
    }
    return "unknown";
}

inline std::ostream &operator<<(std::ostream &out, loop_kind_t kind) {
    out << to_string(kind);
    return out;
}

enum class tile_level_t { thread_group, thread };

class loop_t {
public:
    loop_t() : kind_(loop_kind_t::undef) {}

    loop_t(const expr_t &var, const expr_t &bound, bool is_root)
        : var_(var)
        , kind_(loop_kind_t::serial)
        , bound_(bound)
        , is_root_(is_root) {}

    const expr_t &var() const { return var_; }

    loop_kind_t kind() const { return kind_; }

    void set_kind(loop_kind_t kind) { kind_ = kind; }

    int unroll_factor() const { return unroll_factor_; }

    void set_unroll_factor(int factor) { unroll_factor_ = factor; }

    bool is_kernel_grid() const { return kind() == loop_kind_t::kernel_grid; }

    bool is_serial() const { return kind() == loop_kind_t::serial; }

    bool is_tg_grid() const { return kind() == loop_kind_t::tg_grid; }

    bool is_tensorized() const { return kind() == loop_kind_t::tensorized; }

    const expr_t &bound() const { return bound_; }

    void set_bound(const expr_t &bound) { bound_ = bound; }

    bool is_bound() const { return !bound_var().is_empty(); }

    const expr_t &bound_var() const { return bound_var_; }

    void set_bound_var(const expr_t &v) { bound_var_ = v; }

    bool is_root() const { return is_root_; }

    // Returns true for loops that were neither split, nor fused with other loops.
    bool is_leaf() const { return is_leaf_; }

    // Returns true if this loop was split into outer/inner loops.
    bool is_split_parent() const { return is_split_parent_; }

    // Returns true if this loop was the result of a split.
    bool is_split_child() const { return is_split_child_; }

    // Returns true if this loop was fused with other loops.
    bool is_fused_parent() const { return is_fused_parent_; }

    // Returns true if this loop was the result of a fusion.
    bool is_fused_child() const { return is_fused_child_; }

    const std::vector<expr_t> &parent_vars() const { return parent_vars_; }
    const std::vector<expr_t> &child_vars() const { return child_vars_; }

    void set_split(loop_t &outer_loop, loop_t &inner_loop) {
        outer_loop.parent_vars_.push_back(var());
        child_vars_.push_back(outer_loop.var());
        outer_loop.is_split_child_ = true;

        inner_loop.parent_vars_.push_back(var());
        child_vars_.push_back(inner_loop.var());
        inner_loop.is_split_child_ = true;

        is_split_parent_ = true;
        is_leaf_ = false;
    }

    void set_fuse(std::vector<std::reference_wrapper<loop_t>> &loops) {
        for (auto &l_ref : loops) {
            auto &l = l_ref.get();
            parent_vars_.push_back(l.var());
            l.child_vars_.push_back(var());
            l.is_fused_parent_ = true;
            l.is_leaf_ = false;
        }
        is_fused_child_ = true;
    }

    // Returns a loop variable expressed in the variables of the leaf loops.
    expr_t expand_var(const object_map_t<expr_t, loop_t> &all_loops,
            bool skip_fused = false) const {
        if (is_leaf()) return var();
        if (is_split_parent()) {
            ir_assert(child_vars_.size() == 2);
            auto &outer_loop = all_loops.at(child_vars_[0]);
            auto &inner_loop = all_loops.at(child_vars_[1]);
            auto outer_var = outer_loop.expand_var(all_loops, skip_fused);
            auto inner_var = inner_loop.expand_var(all_loops, skip_fused);
            return outer_var * inner_loop.bound() + inner_var;
        }
        if (is_fused_parent()) {
            if (skip_fused) return var();
            // Example of "unpacking":
            //     fused_var = (a * b * c * d)
            //     b = (fused_var / (D * C)) % B
            ir_assert(child_vars_.size() == 1);
            auto &fused_loop = all_loops.at(child_vars_[0]);
            int nvars = int(fused_loop.parent_vars_.size());
            expr_t denom = 1;
            for (int i = nvars - 1; i >= 0; i--) {
                auto &v = fused_loop.parent_vars_[i];
                auto &child_loop = all_loops.at(v);
                auto &bound = child_loop.bound();
                if (v.is_same(var())) {
                    auto e = fused_loop.expand_var(all_loops, skip_fused)
                            / denom;
                    return (i == 0 ? e : e % bound);
                }
                denom *= bound;
            }
        }

        ir_error_not_expected();
        return expr_t();
    }

    std::string str() const {
        using namespace ir_utils;

        std::ostringstream oss;
        oss << "var: " << var_;
        oss << " bound: " << bound_;
        oss << " kind: " << kind_;
        if (unroll_factor_ != 1) oss << " unroll: " << unroll_factor_;
        std::vector<std::string> props;
        if (is_root()) props.push_back("root");
        if (is_fused_child()) props.push_back("fused");
        if (is_split_parent()) props.push_back("split");
        oss << "(" << make_seq_print_helper(props, ", ") << ")";
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    expr_t var_; // Loop index variable.
    loop_kind_t kind_; // Loop kind.
    expr_t bound_; // Loop bound (exclusive).

    expr_t bound_var_; // External variable this loop bound to.

    int unroll_factor_ = 1;

    bool is_root_ = false;
    bool is_leaf_ = true;

    bool is_split_parent_ = false;
    bool is_split_child_ = false;

    bool is_fused_parent_ = false;
    bool is_fused_child_ = false;

    // For variables there were split or fused.
    // Fusion: i x j -> k
    //     i.child_vars _= [k]
    //     j.child_vars _= [k]
    //     k.parent_vars_ = [i, j]
    // Split: i -> j x k
    //     i.child_vars_ = [j, k]
    //     j.parent_vars_ = [i]
    //     k.parent_vars_ = [i]
    std::vector<expr_t> parent_vars_;
    std::vector<expr_t> child_vars_;
};

// Defines GEMM computation including:
// - Blocking scheme (order of loops, tiles per thread group/thread)
// - Mapping of problem dimensions to GEMM dimensions (BMNK)
class gemm_schedule_t {
public:
    gemm_schedule_t() = default;

    gemm_schedule_t(constraint_set_t &cset, const grid_info_t &kernel_grid,
            const grid_info_t &tg_grid)
        : cset_(&cset), kernel_grid_(kernel_grid), tg_grid_(tg_grid) {}

    const grid_info_t &kernel_grid() const { return kernel_grid_; }
    const grid_info_t &tg_grid() const { return tg_grid_; }

    bmnk_kind_t bmnk_kind(const expr_t &var) const {
        return bmnk_kind(std::vector<expr_t>({var}));
    }

    const bmnk_mapper_t &bmnk_mapper() const { return bmnk_mapper_; }

    void set_b_vars(const std::vector<expr_t> &vars) {
        for (auto &v : vars)
            set_bmnk_kind(v, bmnk_kind_t::b);
    }

    void set_m_vars(const std::vector<expr_t> &vars) {
        for (auto &v : vars)
            set_bmnk_kind(v, bmnk_kind_t::m);
    }

    void set_n_vars(const std::vector<expr_t> &vars) {
        for (auto &v : vars)
            set_bmnk_kind(v, bmnk_kind_t::n);
    }

    void set_k_vars(const std::vector<expr_t> &vars) {
        for (auto &v : vars)
            set_bmnk_kind(v, bmnk_kind_t::k);
    }

    // A/B/C views in the problem notation.
    const view_t &a_view() const { return a_view_; }
    const view_t &b_view() const { return b_view_; }
    const view_t &c_view() const { return c_view_; }

    void set_a_view(const view_t &v) {
        set_view(v, a_view_);
        bmnk_mapper_.set_a_vars(a_view_.vvars());
    }

    void set_b_view(const view_t &v) {
        set_view(v, b_view_);
        bmnk_mapper_.set_b_vars(b_view_.vvars());
    }

    void set_c_view(const view_t &v) {
        set_view(v, c_view_);
        bmnk_mapper_.set_c_vars(c_view_.vvars());
    }

    view_t a_tg_view() const {
        ir_assert(is_finalized_);
        return a_view_.create_sub_view(a_tg_tile_);
    }

    view_t b_tg_view() const {
        ir_assert(is_finalized_);
        return b_view_.create_sub_view(b_tg_tile_);
    }

    view_t c_tg_view() const {
        ir_assert(is_finalized_);
        return c_view_.create_sub_view(c_tg_tile_);
    }

    // Thread group tiles for A, B, C.
    const tensor_t &a_tg_tile() const { return a_tg_tile_; }
    const tensor_t &b_tg_tile() const { return b_tg_tile_; }
    const tensor_t &c_tg_tile() const { return c_tg_tile_; }

    // Thread tiles for A, B, C.
    tensor_t a_thr_tile(bool is_relative = true) const {
        if (is_relative) return a_thr_tile_;
        return a_tg_tile_.create_sub_tensor(a_thr_tile_);
    }

    tensor_t b_thr_tile(bool is_relative = true) const {
        if (is_relative) return b_thr_tile_;
        return b_tg_tile_.create_sub_tensor(b_thr_tile_);
    }

    tensor_t c_thr_tile(bool is_relative = true) const {
        if (is_relative) return c_thr_tile_;
        return c_tg_tile_.create_sub_tensor(c_thr_tile_);
    }

    // Splits loop defined by `var` into two new loops based on `factor`.
    // Before:
    //     for (int var = 0; var < I; var++) { ... }
    // After:
    //   for (int outer_var = 0; outer_var < I / factor; outer_var++) {
    //     for (int inner_var = 0; inner_var < factor; inner_var++) {
    //       ...
    //     }
    //   }
    void split(const expr_t &var, int factor, expr_t &outer_var,
            expr_t &inner_var) {
        auto &loop = find_loop(var);
        ir_assert(loop.is_leaf()) << "Can't split, non-leaf loop.";

        int bound = to_cpp<int>(loop.bound());
        if (loop.is_root() && (bound % factor != 0)) {
            // Auto round-up bounds for the root loops.
            bound = utils::rnd_up(bound, factor);
            loop.set_bound(bound);
        }

        ir_assert(bound % factor == 0) << "Can't split.";

        outer_var = create_var({var}, "outer");
        inner_var = create_var({var}, "inner");
        auto &outer_loop = create_loop(outer_var, bound / factor);
        auto &inner_loop = create_loop(inner_var, factor);
        loop.set_split(outer_loop, inner_loop);
        set_bmnk_kind(outer_var, bmnk_kind(var));
        set_bmnk_kind(inner_var, bmnk_kind(var));
    }

    // Double split.
    void split(const expr_t &var, int factor0, int factor1, expr_t &outer_var0,
            expr_t &outer_var1, expr_t &inner_var) {
        expr_t dummy_inner_var;
        split(var, factor0, outer_var0, dummy_inner_var);
        split(dummy_inner_var, factor1, outer_var1, inner_var);
    }

    // Fuses loops defined by `v0` and `v1` variables, v0 - outer variable, v1
    // - inner variable.
    // Before:
    //   for (int v0 = 0; v0 < V0; v0++) {
    //     for (int v1 = 0; v1 < V1; v1++) { ... }
    //   }
    // After:
    //   for (int v = 0; v < V0 * V1; v++) {
    //       int v0 = v / V1;
    //       int v1 = v % V1;
    //       ...
    //   }
    expr_t fuse(const expr_t &v0, const expr_t &v1) { return fuse({v0, v1}); }

    // Double fuse, v0 - outermost variable, v2 - innermost variable.
    expr_t fuse(const expr_t &v0, const expr_t &v1, const expr_t &v2) {
        return fuse({v0, v1, v2});
    }

    // Fusion of multiple loops.
    expr_t fuse(const std::vector<expr_t> &vars) {
        auto fused_var = create_var(vars, "fused");
        expr_t fused_bound = find_loop(vars[0]).bound();
        for (int i = 1; i < int(vars.size()); i++) {
            auto &loop = find_loop(vars[i]);
            fused_bound *= loop.bound();
        }
        auto &fused_loop = create_loop(fused_var, fused_bound);
        std::vector<std::reference_wrapper<loop_t>> loop_refs;
        for (auto &v : vars) {
            loop_refs.push_back(find_loop(v));
        }
        fused_loop.set_fuse(loop_refs);
        set_bmnk_kind(fused_var, bmnk_kind(vars));
        return fused_var;
    }

    // Sets unrolling factor for the given loop.
    void unroll(const expr_t &v, int factor) {
        auto &loop = find_loop(v);
        loop.set_unroll_factor(factor);
    }

    // Marks the loop defined by `v` as tensorized.
    void tensorize(const expr_t &v) {
        auto &loop = find_loop(v);
        loop.set_kind(loop_kind_t::tensorized);
    }

    // Binds the loop defined by `v` to an external variable.
    void bind(const expr_t &v, const expr_t &bound_var) {
        auto &loop = find_loop(v);
        ir_assert(loop.is_leaf()) << "Can't bind non-leaf loop: " << v;
        loop.set_bound_var(bound_var);
        loop.set_kind(bound_var_to_loop_kind(bound_var));

        int var_dim = bound_var_to_dim(bound_var);
        ir_assert(to_cpp<int>(loop.bound()) == var_dim)
                << "Dimension size doesn't match.";
    }

    // Reorders loops defiend by given variables.
    void reorder(const std::vector<expr_t> &ordered_vars) {
        for (auto &v : ordered_vars) {
            auto &loop = find_loop(v);
            ir_assert(loop.is_leaf()) << "Can't reorder non-leaf loop: " << v;
        }
        std::vector<bool> found(vars_.size());
        for (size_t i = 0; i < vars_.size(); i++) {
            for (size_t j = 0; j < ordered_vars.size(); j++) {
                if (ordered_vars[j].is_same(vars_[i])) {
                    found[i] = true;
                    break;
                }
            }
        }

        for (size_t i = 0, j = 0; i < vars_.size(); i++) {
            if (!found[i]) continue;
            vars_[i] = ordered_vars[j++];
        }
    }

    bool with_thread_group_k_slicing() const {
        ir_assert(is_finalized_);
        dim_t k_thr = 1;
        dim_t k_tg = 1;
        for (int i = 0; i < bmnk_mapper_.ndims(abc_kind_t::a); i++) {
            if (bmnk_mapper_.bmnk_kind(abc_kind_t::a, i) != bmnk_kind_t::k)
                continue;
            k_thr *= a_thr_tile_(i);
            k_tg *= a_tg_tile_(i);
        }
        ir_assert(k_tg % k_thr == 0);
        return k_thr < k_tg;
    }

    void finalize() {
        sort_vars();
        init_problem_tiles();
        init_constraint_set();
        is_finalized_ = true;
    }

    // Returns a statement describing the loop nest of the schedule.
    stmt_t create_loop_nest(const stmt_t &_body = stmt_t()) const {
        stmt_t body = _body;
        for (auto it = vars_.rbegin(); it != vars_.rend(); it++) {
            auto &var = *it;
            auto &loop = find_loop(var);
            if (!loop.is_leaf() || loop.is_tensorized() || loop.is_bound())
                continue;
            body = maybe_inject_let_for_fused_vars(body, loop);
            body = for_t::make(
                    var, 0, loop.bound(), body, loop.unroll_factor());
        }
        return body;
    }

    stmt_t create_bind_stmt(const stmt_t &_body = stmt_t()) const {
        stmt_t body = _body;
        for (auto it = vars_.rbegin(); it != vars_.rend(); it++) {
            auto &var = *it;
            auto &loop = find_loop(var);
            if (!loop.is_leaf() || !loop.is_bound()) continue;
            body = maybe_inject_let_for_fused_vars(body, loop);
            body = let_t::make(var, loop.bound_var(), body);
        }
        return body;
    }

private:
    // Describes split of a root loop into sub-loops.
    class split_info_t {
    public:
        split_info_t(const loop_t *root_loop) : root_loop_(root_loop) {}

        int nloops() const { return int(loops_.size()); }

        void add_sub_loop(
                const loop_t *loop, loop_kind_t loop_kind, int loop_level) {
            loops_.push_back(loop);
            loop_kinds_.push_back(loop_kind);
            loop_levels_.push_back(loop_level);
        }

        // Verifies that sub-loops are ordered from outermost to innermost
        // according to the schedule conventions. There are three set of loops:
        // 1) Loops bound to kernel grid
        // 2) Loops bound to thread group grid and serial loops
        // 3) Tensorized loops
        // Sets of loops must be ordered from outermost to innermost going from
        // 1 to 3. Inside a set loops can be ordered arbitrarily.
        bool is_valid() const {
            auto get_loop_key = [&](int loop_idx) {
                switch (loop_kinds_[loop_idx]) {
                    case loop_kind_t::kernel_grid: return -1;
                    case loop_kind_t::tg_grid:
                    case loop_kind_t::serial: return loop_levels_[loop_idx];
                    case loop_kind_t::tensorized:
                        return std::numeric_limits<int>::max();
                    default: ir_error_not_expected();
                }
                return -1;
            };
            int prev_key = -1;
            for (int i = 0; i < nloops(); i++) {
                int key = get_loop_key(i);
                if (key < prev_key) return false;
                prev_key = key;
            }
            return true;
        }

        // Returns total extent of all loops at a given tile level.
        dim_t dim(tile_level_t tile_level) const {
            dim_t ret = 1;
            for (int i = 0; i < nloops(); i++) {
                switch (loop_kinds_[i]) {
                    case loop_kind_t::kernel_grid:
                    case loop_kind_t::serial: continue;
                    case loop_kind_t::tg_grid:
                        if (tile_level == tile_level_t::thread) continue;
                        break;
                    case loop_kind_t::tensorized: break;
                    default: ir_error_not_expected();
                }
                ret *= to_cpp<dim_t>(loops_[i]->bound());
            }
            return ret;
        }

        // Returns initial offset expressed in the outer variables at a given
        // tile level.
        expr_t start(const object_map_t<expr_t, loop_t> &all_loops,
                tile_level_t tile_level) const {
            auto ret = root_loop_->expand_var(all_loops, /*skip_fused=*/true);
            for (int i = 0; i < nloops(); i++) {
                switch (loop_kinds_[i]) {
                    case loop_kind_t::kernel_grid:
                    case loop_kind_t::serial:
                        if (tile_level == tile_level_t::thread) break;
                        continue;
                    case loop_kind_t::tg_grid:
                        if (tile_level == tile_level_t::thread) continue;
                        break;
                    case loop_kind_t::tensorized: break;
                    default: ir_error_not_expected();
                }
                ret = substitute(ret, loops_[i]->var(), expr_t(0));
            }
            return simplify(ret);
        }

    private:
        const loop_t *root_loop_;
        std::vector<const loop_t *> loops_;
        std::vector<loop_kind_t> loop_kinds_;
        std::vector<int> loop_levels_;
    };

    bmnk_kind_t bmnk_kind(const std::vector<expr_t> &vars) const {
        if (vars.empty()) return bmnk_kind_t::undef;
        if (vars.size() == 1) return bmnk_mapper_.bmnk_kind(vars[0]);
        bmnk_kind_t ret = bmnk_kind(vars[0]);
        for (size_t i = 1; i < vars.size(); i++) {
            if (bmnk_kind(vars[i]) != ret) return bmnk_kind_t::undef;
        }
        return ret;
    }

    void set_bmnk_kind(const expr_t &var, bmnk_kind_t kind) {
        bmnk_mapper_.set_bmnk_kind(var, kind);
    }

    void set_view(const view_t &view, view_t &this_view) {
        this_view = view;
        // Create missing loops.
        for (int i = 0; i < view.nvdims(); i++) {
            auto &v = view.vvars()[i];
            dim_t bound = view.vdims()[i];
            if (has_loop(v)) {
                auto &loop = find_loop(v);
                ir_assert(bound == to_cpp<dim_t>(loop.bound()))
                        << "Inconsistent sizes.";
                continue;
            }
            create_loop(v, bound, /*is_root=*/true);
        }
    }

    loop_kind_t bound_var_to_loop_kind(const expr_t &v) const {
        for (int i = 0; i < kernel_grid_.ndims(); i++) {
            if (kernel_grid_.idx(i).is_same(v)) return loop_kind_t::kernel_grid;
        }
        for (int i = 0; i < tg_grid_.ndims(); i++) {
            if (tg_grid_.idx(i).is_same(v)) return loop_kind_t::tg_grid;
        }
        ir_error_not_expected() << "Unknown external variable: " << v;
        return loop_kind_t::undef;
    }

    int bound_var_to_dim(const expr_t &v) const {
        for (int i = 0; i < kernel_grid_.ndims(); i++) {
            if (kernel_grid_.idx(i).is_same(v)) return kernel_grid_.dim(i);
        }
        for (int i = 0; i < tg_grid_.ndims(); i++) {
            if (tg_grid_.idx(i).is_same(v)) return tg_grid_.dim(i);
        }
        ir_error_not_expected() << "Unknown external variable: " << v;
        return -1;
    }

    bool has_loop(const expr_t &var) const {
        auto it = loops_.find(var);
        return it != loops_.end();
    }

    const loop_t &find_loop(const expr_t &var) const {
        ir_assert(has_loop(var)) << "Var not found: " << var;
        return loops_.at(var);
    }

    loop_t &find_loop(const expr_t &var) {
        ir_assert(has_loop(var)) << "Var not found: " << var;
        return loops_[var];
    }

    int loop_level(const expr_t &var) const {
        for (int i = 0; i < int(vars_.size()); i++) {
            if (vars_[i].is_same(var)) return i;
        }
        return -1;
    }

    loop_t &create_loop(
            const expr_t &var, const expr_t &bound, bool is_root = false) {
        loop_t loop(var, bound, is_root);
        auto ret = loops_.insert({var, loop});
        ir_assert(ret.second) << "Variable already exists: " << var;
        vars_.push_back(var);
        return ret.first->second;
    }

    static std::string strip_suffix(
            const std::string &s, const std::string &suffix) {
        auto pos = s.find(suffix);
        if (pos == std::string::npos) return s;
        if (pos + suffix.length() != s.length()) return s;
        return s.substr(0, pos);
    }

    static expr_t create_var(
            const std::vector<expr_t> &vars, const std::string &suffix) {
        std::string var_name;
        for (auto &v : vars) {
            auto name = strip_suffix(v.as<var_t>().name, "_idx");
            var_name += name + "_";
        }
        var_name += suffix;
        return var_t::make(type_t::s32(), var_name);
    }

    int get_var_key(const expr_t &v) const {
        int key_max = std::numeric_limits<int>::max();
        auto &loop = find_loop(v);
        if (!loop.is_leaf()) return key_max;
        // Loops bound to the kernel grid.
        if (loop.is_kernel_grid()) {
            return kernel_grid_.ndims()
                    - kernel_grid_.dim_idx(loop.bound_var());
        }
        // Loops bound to the thread group grid or serial loop.
        if (loop.is_tg_grid() || loop.is_serial()) return 10;

        // Tensorized loops are the innermost.
        if (loop.is_tensorized()) return key_max - 1;
        ir_error_not_expected() << "Unknown loop";
        return -1;
    }

    void sort_vars() {
        std::stable_sort(vars_.end(), vars_.end(),
                [&](const expr_t &a_var, const expr_t &b_var) {
                    int a_key = get_var_key(a_var);
                    int b_key = get_var_key(b_var);
                    return a_key < b_key;
                });
    }

    void init_problem_tiles() {
        object_map_t<expr_t, split_info_t> split_infos;
        for (auto *view : {&a_view_, &b_view_, &c_view_}) {
            for (auto &v : view->vvars()) {
                if (split_infos.count(v) > 0) continue;
                split_infos.insert({v, get_split_info(v)});
            }
        }
        a_tg_tile_ = compute_problem_tile(
                a_view_.vvars(), split_infos, tile_level_t::thread_group);
        b_tg_tile_ = compute_problem_tile(
                b_view_.vvars(), split_infos, tile_level_t::thread_group);
        c_tg_tile_ = compute_problem_tile(
                c_view_.vvars(), split_infos, tile_level_t::thread_group);
        a_thr_tile_ = compute_problem_tile(
                a_view_.vvars(), split_infos, tile_level_t::thread);
        b_thr_tile_ = compute_problem_tile(
                b_view_.vvars(), split_infos, tile_level_t::thread);
        c_thr_tile_ = compute_problem_tile(
                c_view_.vvars(), split_infos, tile_level_t::thread);
    }

    void init_constraint_set() {
        for (auto &v : vars_) {
            auto &loop = find_loop(v);
            if (loop.is_fused_parent()) {
                cset_->add_constraint(v >= 0);
                cset_->add_constraint(v < loop.bound());
                continue;
            }
            if (!loop.is_leaf()) continue;

            // Fused variables are used only to initialize fused parents.
            if (loop.is_fused_child()) continue;

            if (loop.is_bound()) {
                cset_->add_constraint(v == loop.bound_var());
                continue;
            }

            cset_->add_constraint(v >= 0);
            cset_->add_constraint(v < loop.bound());
        }
    }

    split_info_t get_split_info(const expr_t &root_var) const {
        split_info_t ret(&find_loop(root_var));
        std::function<void(const expr_t &)> walk_down;
        walk_down = [&](const expr_t &v) {
            auto &loop = find_loop(v);
            if (loop.is_leaf() || loop.is_fused_parent()) {
                // Treat a fused var as leaf as it can't be split into other
                // vars.
                loop_kind_t kind = loop.kind();
                int level;
                if (loop.is_fused_parent()) {
                    auto &child_var = loop.child_vars()[0];
                    ir_assert(find_loop(child_var).is_leaf());
                    kind = find_loop(child_var).kind();
                    level = loop_level(child_var);
                } else {
                    level = loop_level(v);
                }
                ret.add_sub_loop(&loop, kind, level);
            } else if (loop.is_split_parent()) {
                walk_down(loop.child_vars()[0]);
                walk_down(loop.child_vars()[1]);
            } else {
                ir_error_not_expected();
            }
        };
        walk_down(root_var);
        ir_assert(ret.is_valid()) << "Invalid loop nest.";
        return ret;
    }

    tensor_t compute_problem_tile(const std::vector<expr_t> &vars,
            const object_map_t<expr_t, split_info_t> &split_infos,
            tile_level_t tile_level) {
        std::vector<dim_t> tile_dims;
        std::vector<expr_t> tile_start;
        for (auto &v : vars) {
            auto &split_info = split_infos.at(v);
            tile_dims.push_back(split_info.dim(tile_level));
            tile_start.push_back(split_info.start(loops_, tile_level));
        }
        return tensor_t(tile_dims, tile_start);
    }

    stmt_t maybe_inject_let_for_fused_vars(
            const stmt_t &_body, const loop_t &loop) const {
        auto body = _body;
        if (!loop.is_leaf() || !loop.is_fused_child()) return body;
        auto &pvars = loop.parent_vars();
        for (auto it = pvars.rbegin(); it != pvars.rend(); it++) {
            auto &ploop = find_loop(*it);
            body = let_t::make(*it, ploop.expand_var(loops_), body);
        }
        return body;
    }

    bool is_finalized_ = false;

    constraint_set_t *cset_;
    grid_info_t kernel_grid_;
    grid_info_t tg_grid_;

    // Loop indices, ordered from outermost to innermost.
    std::vector<expr_t> vars_;

    object_map_t<expr_t, loop_t> loops_;

    bmnk_mapper_t bmnk_mapper_;

    // Full views for A, B, C.
    view_t a_view_;
    view_t b_view_;
    view_t c_view_;

    // Thread group tiles for A, B, C.
    tensor_t a_tg_tile_;
    tensor_t b_tg_tile_;
    tensor_t c_tg_tile_;

    // Thread tiles for A, B, C (relative to thread group tiles).
    tensor_t a_thr_tile_;
    tensor_t b_thr_tile_;
    tensor_t c_thr_tile_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
