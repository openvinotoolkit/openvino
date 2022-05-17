/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef GPU_JIT_GEMM_GEN_GEMM_KERNEL_HPP
#define GPU_JIT_GEMM_GEN_GEMM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gen_gemm_kernel_generator.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/jit/jit_generator_base.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct gen_gemm_kernel_t : public jit_generator_base {

    status_t init_gemm(compute::gpu_arch_t arch) {
        hw_ = convert_dnnl_arch_to_hw(arch);
        if (hw_ == ngen::HW::Unknown) return status::unimplemented;

        auto status = complete_strategy();
        if (status != status::success) return status;

        return init_interface();
    }

    const char *kernel_name() const override { return "gemm_kernel"; }
    cl_kernel get_kernel(cl_context context, cl_device_id device) override;

    CommonDriverInfo driver_info() const;

protected:
    static Type convert_dnnl_to_kernel_type(data_type_t type) {
        switch (type) {
            default: assert(!"Unknown type");
            case data_type::f32: return Type::f32;
            case data_type::f16: return Type::f16;
            case data_type::bf16: return Type::bf16;
            case data_type::s32: return Type::s32;
            case data_type::u8: return Type::u8;
            case data_type::s8: return Type::s8;
        }
    }

    static ngen::HW convert_dnnl_arch_to_hw(compute::gpu_arch_t arch) {
        switch (arch) {
            case compute::gpu_arch_t::gen9: return ngen::HW::Gen9;
            case compute::gpu_arch_t::xe_lp: return ngen::HW::XeLP;
            case compute::gpu_arch_t::xe_hp: return ngen::HW::XeHP;
            case compute::gpu_arch_t::xe_hpg: return ngen::HW::XeHPG;
            default: return ngen::HW::Unknown;
        }
    }

    ngen::HW hw_ = ngen::HW::Unknown;
    GEMMProblem problem_;
    GEMMStrategy strategy_;
    ngen::NEOInterfaceHandler interface_ {ngen::HW::Unknown};
    char strategy_tag_ = '\0';

private:
    static bool matching_hw(ngen::HW hw, ngen::HW hw_ref);
    status_t complete_strategy();
    status_t read_strategy(const char *str);
    status_t init_interface();
};

struct gen_gemm_nocopy_kernel_t : public gen_gemm_kernel_t {
    status_t init(compute::gpu_arch_t arch, int batch_dims, bool trans_a,
            bool trans_b, bool ab_offset, bool c_offset, bool bias,
            const post_ops_t &post_ops, data_type_t a_type, data_type_t b_type,
            data_type_t c_type, data_type_t co_type, data_type_t acc_type,
            int align_a, int align_b, int align_c, int unroll_m, int unroll_n,
            char tag) {

        align_a = nstl::max(align_a, int(types::data_type_size(a_type)));
        align_b = nstl::max(align_b, int(types::data_type_size(b_type)));
        align_c = nstl::max(align_c, int(types::data_type_size(c_type)));

        problem_.Ta = convert_dnnl_to_kernel_type(a_type);
        problem_.Tb = convert_dnnl_to_kernel_type(b_type);
        problem_.Tc = convert_dnnl_to_kernel_type(acc_type);
        problem_.Tco = convert_dnnl_to_kernel_type(co_type);
        problem_.Tc_ext = convert_dnnl_to_kernel_type(c_type);
        problem_.Ts = problem_.Tc;
        problem_.A.layout = trans_a ? MatrixLayout::T : MatrixLayout::N;
        problem_.B.layout = trans_b ? MatrixLayout::T : MatrixLayout::N;
        problem_.C.layout = MatrixLayout::N;
        problem_.A.crosspack = problem_.B.crosspack = problem_.C.crosspack = 1;
        problem_.A.packSize = problem_.B.packSize = problem_.C.packSize = 0;
        problem_.A.padded = problem_.B.padded = problem_.C.padded = false;
        problem_.A.setAlignment(align_a);
        problem_.B.setAlignment(align_b);
        problem_.C.setAlignment(align_c);
        problem_.A.base = ngen::AddressBase::createA64(true);
        problem_.B.base = ngen::AddressBase::createA64(true);
        problem_.C.base = ngen::AddressBase::createA64(true);
        if (batch_dims > 0) {
            problem_.batch = BatchMode::Strided;
            problem_.batchDims = batch_dims;
        }
        if (ab_offset) problem_.abOffset = ABOffset::Calc;
        if (c_type == data_type::s32) problem_.Ts = Type::f32;
        if (post_ops.len() > 0) {
            problem_.post_ops = post_ops;
            if (a_type == data_type::f16) problem_.Ts = Type::f32;
        }
        if (c_offset || bias) {
            assert(!(c_offset && bias));
            problem_.cOffset = bias ? COffset::Pre : COffset::Post;
            problem_.CO.base = ngen::AddressBase::createBTS(0);
            problem_.CO.crosspack = 1;
            problem_.CO.padded = false;
            problem_.CO.alignment = problem_.C.alignment;
        }

        strategy_.unroll[LoopM] = unroll_m;
        strategy_.unroll[LoopN] = unroll_n;
        strategy_tag_ = tag;

        return init_gemm(arch);
    }

    static void choose_unrolls(compute::gpu_arch_t arch, int hw_threads,
            bool trans_a, bool trans_b, data_type_t a_type, data_type_t b_type,
            data_type_t c_type, int align_a, int align_b, int align_c, dim_t m,
            dim_t n, dim_t k, dim_t batch, int batch_dims, int &unroll_m,
            int &unroll_n, char &tag);
};

struct gen_gemm_xehp_systolic_kernel_t : public gen_gemm_kernel_t {
    enum class offset_t { none, fixed, row, column, runtime };

    status_t init(compute::gpu_arch_t arch, int batch_dims, offset_t a_offset,
            offset_t b_offset, offset_t c_offset, offset_t bias, float alpha,
            float beta, const post_ops_t &post_ops, data_type_t a_type,
            data_type_t b_type, data_type_t c_type, data_type_t co_type,
            data_type_t acc_type, int unroll_m, int unroll_n, char tag) {

        bool arch_ok = (arch == compute::gpu_arch_t::xe_hp)
                || (arch == compute::gpu_arch_t::xe_hpg);

        if (!arch_ok) return status::unimplemented;

        auto ksys = int(32 / types::data_type_size(a_type));
        auto csys = int(4 / types::data_type_size(a_type));

        problem_.Ta = convert_dnnl_to_kernel_type(a_type);
        problem_.Tb = convert_dnnl_to_kernel_type(b_type);
        problem_.Tc = convert_dnnl_to_kernel_type(acc_type);
        problem_.Tco = convert_dnnl_to_kernel_type(co_type);
        problem_.Tc_ext = convert_dnnl_to_kernel_type(c_type);
        problem_.Ts = Type::f32;
        problem_.A.layout = MatrixLayout::PackedColumns;
        problem_.B.layout = MatrixLayout::PackedRows;
        problem_.C.layout = MatrixLayout::N;
        problem_.A.crosspack = csys;
        problem_.B.crosspack = ksys;
        problem_.C.crosspack = 1;
        problem_.A.packSize = unroll_m;
        problem_.B.packSize = unroll_n;
        problem_.C.packSize = 0;
        problem_.A.tileR = 8;
        problem_.A.tileC = ksys;
        problem_.A.padded = problem_.B.padded = true;
        problem_.C.padded = false;
        problem_.A.setAlignment(32);
        problem_.B.setAlignment(32);
        problem_.C.setAlignment(int(types::data_type_size(c_type)));
        problem_.A.base = ngen::AddressBase::createA64(true);
        problem_.B.base = ngen::AddressBase::createA64(true);
        problem_.C.base = ngen::AddressBase::createA64(true);
        if (batch_dims > 0) {
            problem_.batch = BatchMode::Strided;
            problem_.batchDims = batch_dims;
        }
        if (a_offset == offset_t::fixed && b_offset == offset_t::fixed)
            problem_.abOffset = ABOffset::Load;
        else if (a_offset != offset_t::none || b_offset != offset_t::none)
            return status::unimplemented;
        if (alpha == 1.0f) problem_.alpha_real = alpha;
        if (beta == 0.0f || beta == 1.0f) problem_.beta_real = beta;
        if (post_ops.len() > 0) {
            problem_.post_ops = post_ops;
            problem_.Ts = Type::f32;
        }
        if (c_offset == offset_t::runtime)
            problem_.cOffset = COffset::Post;
        else if (c_offset != offset_t::none)
            return status::unimplemented;

        if (bias == offset_t::runtime) {
            if (problem_.cOffset != COffset::None) return status::unimplemented;
            problem_.cOffset = COffset::Pre;
        } else if (bias != offset_t::none)
            return status::unimplemented;

        if (problem_.cOffset != COffset::None) {
            problem_.CO.base = ngen::AddressBase::createBTS(0);
            problem_.CO.crosspack = 1;
            problem_.CO.padded = false;
            problem_.CO.alignment = problem_.C.alignment;
        }

        strategy_.unroll[LoopM] = unroll_m;
        strategy_.unroll[LoopN] = unroll_n;
        strategy_tag_ = tag;

        return init_gemm(arch);
    }

    static void choose_unrolls(compute::gpu_arch_t arch, int eu_count,
            data_type_t a_type, data_type_t b_type, data_type_t c_type, dim_t m,
            dim_t n, dim_t k, dim_t batch, int &unroll_m, int &unroll_n,
            char &tag);

    static int min_block_k(data_type_t a_type) { return 2048; }
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
