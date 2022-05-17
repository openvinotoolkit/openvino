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

#ifndef GPU_JIT_CONV_REORDER_SUPPORT_HPP
#define GPU_JIT_CONV_REORDER_SUPPORT_HPP

#include <array>
#include <memory>
#include <string>

#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper class to permute registers. Used to restore registers after applying
// DPAS -> DPASW transformation.
class grf_permutator_t {
public:
    grf_permutator_t(ngen::HW hw, const expr_t &grf_buf_base = expr_t())
        : hw_(hw), grf_buf_base_(grf_buf_base) {
        permutation_.fill(-1);
    }

    const expr_t &grf_buf_base() const { return grf_buf_base_; }

    int map(int base) const {
        ir_assert(grf_base_ != -1) << "GRF base not bound.";
        ir_assert(base >= 0 && base < max_regs);
        if (permutation_[base] == -1) return base;
        return permutation_[base];
    }

    bool is_empty() const {
        if (grf_buf_base_.is_empty()) return true;
        for (int i = 0; i < int(permutation_.size()); i++) {
            if (permutation_[i] != -1 && permutation_[i] != i) return false;
        }
        return true;
    }

    void set_permute(const expr_t &old_grf, const expr_t &new_grf) {
        auto &old_base = old_grf.as<ptr_t>().base;
        auto &new_base = new_grf.as<ptr_t>().base;
        ir_assert(old_base.is_same(grf_buf_base_));
        ir_assert(new_base.is_same(grf_buf_base_));

        int old_off = to_cpp<int>(old_grf.as<ptr_t>().off);
        int new_off = to_cpp<int>(new_grf.as<ptr_t>().off);

        const int grf_size = ngen::GRF::bytes(hw_);

        ir_assert(old_off % grf_size == 0)
                << "Must be aligned to GRF boundary.";
        ir_assert(new_off % grf_size == 0)
                << "Must be aligned to GRF boundary.";

        old_off /= grf_size;
        new_off /= grf_size;

        ir_assert(permutation_[old_off] == -1) << "Already assigned.";
        permutation_[old_off] = new_off;
    }

    void set_grf_base(int grf_base) {
        ir_assert(grf_base_ == -1) << "Can't set GRF base twice.";
        grf_base_ = grf_base;
        // Update offsets.
        auto old_perm = permutation_;
        permutation_.fill(-1);
        for (int i = 0; i < (int)old_perm.size(); i++) {
            if (old_perm[i] != -1 && old_perm[i] != i) {
                ir_assert(grf_base + i < int(permutation_.size()));
                permutation_[grf_base + i] = grf_base + old_perm[i];
            }
        }
    }

private:
    static const int max_regs = 256;

    ngen::HW hw_;
    int grf_base_ = -1;
    expr_t grf_buf_base_;
    std::array<int, max_regs> permutation_;
};

// Implements reorder between GRF buffers in given layouts. Conversion between
// data types is supported.
class reorder_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(reorder_t, func_impl_t)

    static func_t make(const layout_t &src_layout, const layout_t &dst_layout,
            const std::shared_ptr<grf_permutator_t> &grf_perm = nullptr) {
        return func_t(new reorder_t(src_layout, dst_layout, grf_perm));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (src_layout == other.src_layout)
                && (dst_layout == other.dst_layout)
                && (grf_perm.get() == other.grf_perm.get());
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(src_layout, dst_layout, grf_perm.get());
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << "reorder[" << src_layout << ", " << dst_layout << "]";
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst_buf, 0)
    IR_DEFINE_ARG_GET(src_buf, 1)

    layout_t src_layout;
    layout_t dst_layout;

    std::shared_ptr<grf_permutator_t> grf_perm;

private:
    reorder_t(const layout_t &src_layout, const layout_t &dst_layout,
            const std::shared_ptr<grf_permutator_t> &grf_perm)
        : src_layout(src_layout), dst_layout(dst_layout), grf_perm(grf_perm) {}
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
