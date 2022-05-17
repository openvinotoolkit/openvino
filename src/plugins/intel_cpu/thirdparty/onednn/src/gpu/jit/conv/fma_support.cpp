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

#include "gpu/jit/conv/fma_support.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::string fma_kind::to_string(fma_kind_t val) {
    switch (val) {
        case fma_kind_t::mad: return "mad";
        case fma_kind_t::dpas: return "dpas";
        case fma_kind_t::dpasw: return "dpasw";
        case fma_kind_t::unknown: return "unknown";
        default: assert(!"unknown fma kind"); return "unknown";
    }
}

fma_kind_t fma_kind::from_string(std::string enum_string) {
    for (int enum_int = static_cast<int>(fma_kind_t::mad);
            enum_int <= static_cast<int>(fma_kind_t::unknown); enum_int++) {
        fma_kind_t enum_val = static_cast<fma_kind_t>(enum_int);
        if (fma_kind::to_string(enum_val).compare(enum_string) == 0)
            return enum_val;
    }
    assert(!"unknown fma kind");
    return fma_kind_t::unknown;
}

fma_kind_t fma_kind::get_supported_kind(
        ngen::HW hw, const type_t &a, const type_t &b, const type_t &c) {
    if (hw >= ngen::HW::XeHP && dpas_t::matches_types(hw, a, b, c)) {
        return fma_kind_t::dpasw;
    }
    if (mad_t::matches_types(hw, a, b, c)) return fma_kind_t::mad;
    return fma_kind_t::unknown;
}

int fma_kind::get_simd_size(ngen::HW hw, const fma_kind_t kind, const type_t &a,
        const type_t &b, const type_t &c) {
    switch (kind) {
        case fma_kind_t::dpasw:
        case fma_kind_t::dpas: return 8;
        case fma_kind_t::mad: return mad_t::get_simd_size(a, b, c);
        default: return 0;
    }
}

type_t multiply_desc_t::get_c_type(
        const type_t &a, const type_t &b, bool force_c_upconvert) {
    if (utils::one_of(
                a, type_t::s8(), type_t::u8(), type_t::s16(), type_t::s32())
            && utils::one_of(b, type_t::s8(), type_t::u8(), type_t::s16(),
                    type_t::s32()))
        return type_t::s32();

    if (a == type_t::bf16() && b == type_t::bf16()) return type_t::f32();
    if (a == type_t::f32() && b == type_t::f32()) return type_t::f32();

    if (utils::one_of(a, type_t::f16(), type_t::bf16()) && b == type_t::f32()) {
        return type_t::f32();
    }

    if (a == type_t::f16() && b == type_t::f16()) {
        if (force_c_upconvert) return type_t::f32();
        return type_t::f16();
    }

    ir_error_not_expected()
            << "Can't deduce C type. A type: " << a << " B type: " << b;
    return type_t::undef();
}

layout_t dpas_t::a_layout() const {
    if (simd_size == 8) {
        if (src1_type.size() == 1) return layout_t(src1_type, 0, "8b8a4b");
        if (src1_type.size() == 2) return layout_t(src1_type, 0, "8b8a2b");
    }
    if (simd_size == 16) {
        if (src1_type.size() == 1) return layout_t(src1_type, 0, "8b16a4b");
        if (src1_type.size() == 2) return layout_t(src1_type, 0, "8b16a2b");
    }
    ir_error_not_expected();
    return layout_t();
}

layout_t dpas_t::b_layout() const {
    if (src2_type.size() != 1 && src2_type.size() != 2) ir_error_not_expected();

    dim_t blk = src2_type.size() == 1 ? 32 : 16;
    std::vector<dim_t> dims = {rcount, blk};
    return layout_t(src2_type, 0, dims).transpose();
}

layout_t dpas_t::c_layout() const {
    std::vector<dim_t> dims = {rcount, simd_size};
    return layout_t(dst_type, 0, dims).transpose();
}

bool dpas_t::matches(const multiply_desc_t &desc) const {
    int m_blk = simd_size;
    int n_blk = rcount;
    int k_blk = sdepth * 4 / src1_type.size();

    if (desc.m() % m_blk != 0 || desc.n() % n_blk != 0 || desc.k() % k_blk != 0)
        return false;

    auto a_blk_layout = desc.a_layout().map(tensor_t({m_blk, k_blk}));
    auto b_blk_layout = desc.b_layout().map(tensor_t({k_blk, n_blk}));

    if (a_blk_layout != a_layout()) return false;
    if (b_blk_layout != b_layout()) return false;

    return true;
}

bool dpas_t::matches_types(
        ngen::HW hw, const type_t &a, const type_t &b, const type_t &c) {
    if (a.is_x8() && b.is_x8() && c.is_s32()) return true;
    if (a.is_f16() && b.is_f16() && c.is_f32()) return true;
    if (a.is_bf16() && b.is_bf16() && c.is_f32()) return true;

    return false;
}

bool mad_t::matches_types(
        ngen::HW hw, const type_t &a, const type_t &b, const type_t &c) {
    if (a != b) return false;

    if (a.is_f32() && c.is_f32()) return true;
    if (a.is_f16() && c.is_f16()) return true;
    if (a.is_f16() && c.is_f32()) return true;
    if (hw >= ngen::HW::XeHP) {
        if (a.is_bf16() && c.is_f32()) return true;
        if (a.is_f32() && c.is_bf16()) return true;
    }
    if (a.is_x8() && c.is_x16()) return true;
    if ((a.is_x16() || a.is_x32()) && (c.is_x16() || c.is_x32())) return true;

    return false;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
