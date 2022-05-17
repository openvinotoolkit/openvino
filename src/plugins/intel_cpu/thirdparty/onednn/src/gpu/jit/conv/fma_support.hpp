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

#ifndef GPU_JIT_CONV_FMA_SUPPORT_HPP
#define GPU_JIT_CONV_FMA_SUPPORT_HPP

#include <sstream>
#include <string>

#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Possible backend instruction sets
enum class fma_kind_t {
    mad,
    dpas,
    dpasw,
    unknown,
};

namespace fma_kind {

std::string to_string(fma_kind_t val);
fma_kind_t from_string(std::string enum_string);

fma_kind_t get_supported_kind(
        ngen::HW hw, const type_t &a, const type_t &b, const type_t &c);

int get_simd_size(ngen::HW hw, fma_kind_t kind, const type_t &a,
        const type_t &b, const type_t &c);

} // namespace fma_kind

class multiply_desc_t {
public:
    multiply_desc_t() = default;

    multiply_desc_t(const layout_t &a_layout, const layout_t &b_layout,
            bool force_c_upconvert)
        : a_layout_(a_layout), b_layout_(b_layout) {
        ir_assert(a_layout.ndims() == 2 && b_layout.ndims() == 2)
                << "Expected 2D layouts, A layout: " << a_layout
                << " B layout: " << b_layout;

        c_type_ = get_c_type(a_type(), b_type(), force_c_upconvert);
    }

    const layout_t &a_layout() const { return a_layout_; }
    const layout_t &b_layout() const { return b_layout_; }

    const type_t &a_type() const { return a_layout_.type(); }
    const type_t &b_type() const { return b_layout_.type(); }
    const type_t &c_type() const { return c_type_; }

    int m() const { return a_layout_.dims()[0]; }
    int n() const { return b_layout_.dims()[1]; }
    int k() const { return a_layout_.dims()[1]; }

    static type_t get_c_type(
            const type_t &a, const type_t &b, bool force_c_upconvert);

private:
    layout_t a_layout_;
    layout_t b_layout_;
    type_t c_type_;
};

// Function representing DPAS instruction.
class dpas_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(dpas_t, func_impl_t)

    static func_t make(bool is_dpasw, int simd_size, int sdepth, int rcount,
            const type_t &dst_type, const type_t &src1_type,
            const type_t &src2_type) {
        return func_t(new dpas_t(is_dpasw, simd_size, sdepth, rcount, dst_type,
                src1_type, src2_type));
    }

    static func_t make_dpasw(const dpas_t &dpas) {
        return func_t(new dpas_t(true, dpas.simd_size, dpas.sdepth, dpas.rcount,
                dpas.dst_type, dpas.src1_type, dpas.src2_type));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (is_dpasw == other.is_dpasw) && (sdepth == other.sdepth)
                && (rcount == other.rcount) && (dst_type == other.dst_type)
                && (src1_type == other.src1_type)
                && (src2_type == other.src2_type);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(
                is_dpasw, sdepth, rcount, dst_type, src1_type, src2_type);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << (is_dpasw ? "dpasw" : "dpas");
        oss << "." << sdepth << "x" << rcount;
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst, 0)
    IR_DEFINE_ARG_GET(src0, 1)
    IR_DEFINE_ARG_GET(src1, 2)
    IR_DEFINE_ARG_GET(src2, 3)

    stmt_t operator()(const expr_t &dst, const expr_t &src0, const expr_t &src1,
            const expr_t &src2) const {
        return call({dst, src0, src1, src2});
    }

    int dst_size() const { return simd_size * rcount * sizeof(uint32_t); }
    int src0_size() const { return dst_size(); }
    int src1_size() const { return simd_size * sdepth * sizeof(uint32_t); }
    int src2_size() const {
        const int dpas_size = sdepth * rcount * sizeof(uint32_t);
        return is_dpasw ? dpas_size / 2 : dpas_size;
    }

    layout_t a_layout() const;
    layout_t b_layout() const;
    layout_t c_layout() const;

    bool matches(const multiply_desc_t &desc) const;

    static bool matches_types(
            ngen::HW hw, const type_t &a, const type_t &b, const type_t &c);

    bool is_dpasw;

    int simd_size;
    int sdepth;
    int rcount;

    type_t dst_type; // src0 type is same as dst_type.
    type_t src1_type;
    type_t src2_type;

private:
    dpas_t(bool is_dpasw, int simd_size, int sdepth, int rcount,
            const type_t &dst_type, const type_t &src1_type,
            const type_t &src2_type)
        : is_dpasw(is_dpasw)
        , simd_size(simd_size)
        , sdepth(sdepth)
        , rcount(rcount)
        , dst_type(dst_type)
        , src1_type(src1_type)
        , src2_type(src2_type) {}
};

// Function representing MAD instruction.
class mad_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(mad_t, func_impl_t)

    static func_t make(const type_t &dst_type, int simd_size,
            const type_t &src1_type, int src1_stride, const type_t src2_type,
            int src2_stride) {
        return func_t(new mad_t(dst_type, simd_size, src1_type, src1_stride,
                src2_type, src2_stride));
    }

    bool is_equal(const object_impl_t *obj) const override {
        if (!obj->is<self_type>()) return false;
        auto &other = obj->as<self_type>();

        return (dst_type == other.dst_type) && (src1_type == other.src1_type)
                && (src2_type == other.src2_type)
                && (simd_size == other.simd_size)
                && (src1_stride == other.src1_stride)
                && (src2_stride == other.src2_stride);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(dst_type, src1_type, src2_type, simd_size,
                src2_stride, src1_stride);
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << "mad";
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst, 0)
    IR_DEFINE_ARG_GET(src0, 1)
    IR_DEFINE_ARG_GET(src1, 2)
    IR_DEFINE_ARG_GET(src2, 3)

    stmt_t operator()(const expr_t &dst, const expr_t &src0, const expr_t &src1,
            const expr_t &src2) const {
        return call({dst, src0, src1, src2});
    }

    int dst_size() const { return simd_size * dst_type.size(); }
    int src0_size() const { return dst_size(); }
    int src1_size() const {
        return std::max(
                src1_type.size(), src1_stride * src1_type.size() * simd_size);
    }
    int src2_size() const {
        return std::max(
                src2_type.size(), src2_stride * src2_type.size() * simd_size);
    }

    static bool matches_types(
            ngen::HW hw, const type_t &a, const type_t &b, const type_t &c);

    static const int max_exec_size = 32;
    static const int max_exec_size_bytes = 64;
    static int get_simd_size(
            const type_t &a, const type_t &b, const type_t &c) {
        int max_size = max_exec_size;
        if (max_exec_size_bytes / a.size() < max_size)
            max_size = max_exec_size_bytes / a.size();
        if (max_exec_size_bytes / b.size() < max_size)
            max_size = max_exec_size_bytes / b.size();
        if (max_exec_size_bytes / c.size() < max_size)
            max_size = max_exec_size_bytes / c.size();
        return max_size;
    }
    int get_simd_size() const { return simd_size; }

    type_t dst_type;
    type_t src1_type;
    type_t src2_type;

    int simd_size;
    int src1_stride;
    int src2_stride;

private:
    mad_t(const type_t &dst_type, int simd_size, const type_t &src1_type,
            int src1_stride, const type_t &src2_type, int src2_stride)
        : dst_type(dst_type)
        , src1_type(src1_type)
        , src2_type(src2_type)
        , simd_size(simd_size)
        , src1_stride(src1_stride)
        , src2_stride(src2_stride) {

        ir_assert(math::is_pow2(simd_size));

        ir_assert(simd_size <= max_exec_size);
        ir_assert(dst_size() <= max_exec_size_bytes);
        ir_assert(src1_size() <= max_exec_size_bytes);
        ir_assert(src2_size() <= max_exec_size_bytes);
    }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
