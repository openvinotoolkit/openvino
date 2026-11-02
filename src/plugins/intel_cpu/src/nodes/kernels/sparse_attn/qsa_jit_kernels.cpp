// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_jit_kernels.hpp"

#include <cache/multi_cache.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "nodes/kernels/sparse_attn/qsa_score_kernel.hpp"
#include "nodes/kernels/x64/jit_kernel_base.hpp"
#include "nodes/kernels/x64/rms_kernel.hpp"
#include "nodes/kernels/x64/rope_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node::qsa {

namespace {

using dnnl::impl::cpu::x64::avx2;
using dnnl::impl::cpu::x64::avx512_core;
using dnnl::impl::cpu::x64::mayiuse;

std::shared_ptr<kernel::JitKernelBase> make_rms_kernel(const RmsRopeKey& key) {
    kernel::jit_rms_compile_params jcp;
    jcp.src_prc = ov::element::f32;
    jcp.dst_prc = ov::element::f32;
    jcp.data_size = key.dim;
    jcp.scale_size = key.dim;
    jcp.eps = key.eps;
    std::shared_ptr<kernel::JitKernelBase> res;
    if (mayiuse(avx512_core)) {
        res = std::make_shared<kernel::jit_rms_kernel<avx512_core>>(jcp);
    } else if (mayiuse(avx2)) {
        res = std::make_shared<kernel::jit_rms_kernel<avx2>>(jcp);
    }
    if (res) {
        res->create_kernel();
    }
    return res;
}

std::shared_ptr<kernel::JitKernelBase> make_rotary_kernel(const RmsRopeKey& key) {
    kernel::jit_rotary_compile_params jcp;
    jcp.src_prc = ov::element::f32;
    jcp.dst_prc = ov::element::f32;
    jcp.rotary_ndims = key.rotary_dim;
    jcp.cos_sin_ndims = key.rotary_dim;
    jcp.mode = kernel::jit_rotary_compile_params::Mode::ROTATE_HALF;
    std::shared_ptr<kernel::JitKernelBase> res;
    if (mayiuse(avx512_core)) {
        res = std::make_shared<kernel::jit_rotary_kernel<avx512_core>>(jcp);
    } else if (mayiuse(avx2)) {
        res = std::make_shared<kernel::jit_rotary_kernel<avx2>>(jcp);
    }
    if (res) {
        res->create_kernel();
    }
    return res;
}

}  // namespace

std::shared_ptr<const JitRmsRope> JitRmsRope::get(const MultiCachePtr& cache, const RmsRopeKey& key) {
    const auto builder = [](const RmsRopeKey& k) -> std::shared_ptr<const JitRmsRope> {
        auto rms = make_rms_kernel(k);
        auto rotary = k.rotary_dim > 0 ? make_rotary_kernel(k) : nullptr;
        if (!rms || (k.rotary_dim > 0 && !rotary)) {
            return nullptr;
        }
        return std::make_shared<const JitRmsRope>(std::move(rms), std::move(rotary));
    };
    try {
        return cache->getOrCreate(key, builder).first;
    } catch (const ov::Exception&) {
        return nullptr;
    }
}

JitRmsRope::JitRmsRope(std::shared_ptr<kernel::JitKernelBase> rms, std::shared_ptr<kernel::JitKernelBase> rotary)
    : m_rms(std::move(rms)),
      m_rotary(std::move(rotary)) {}

void JitRmsRope::operator()(const float* src,
                            float* dst,
                            const float* weight,
                            const float* cos,
                            const float* sin) const {
    kernel::jit_rms_call_args rms_args{reinterpret_cast<const uint8_t*>(src), weight, reinterpret_cast<uint8_t*>(dst)};
    (*m_rms)(&rms_args);
    if (m_rotary) {
        kernel::jit_rotary_call_args rotary_args{dst, cos, sin, dst};
        (*m_rotary)(&rotary_args);
    }
}

bool JitQsaScore::supports(const ov::element::Type& row_prc, size_t dim) {
    if (row_prc != ov::element::f32 && row_prc != ov::element::bf16 && row_prc != ov::element::f16) {
        return false;
    }
    size_t vec_size = 0;
    if (mayiuse(avx512_core)) {
        vec_size = kernel::jit_qsa_score_kernel<avx512_core>::vec_size;
    } else if (mayiuse(avx2)) {
        vec_size = kernel::jit_qsa_score_kernel<avx2>::vec_size;
    }
    return vec_size != 0 && dim >= vec_size && dim % vec_size == 0;
}

std::shared_ptr<const JitQsaScore> JitQsaScore::get(const MultiCachePtr& cache, const QsaScoreKey& key) {
    if (!supports(key.row_prc, key.dim) || key.heads == 0) {
        return nullptr;
    }
    const auto builder = [](const QsaScoreKey& k) -> std::shared_ptr<const JitQsaScore> {
        kernel::jit_qsa_score_compile_params jcp;
        jcp.row_prc = k.row_prc;
        jcp.dim = k.dim;
        jcp.heads = k.heads;
        std::shared_ptr<kernel::JitKernelBase> res;
        if (mayiuse(avx512_core)) {
            res = std::make_shared<kernel::jit_qsa_score_kernel<avx512_core>>(jcp);
        } else {
            res = std::make_shared<kernel::jit_qsa_score_kernel<avx2>>(jcp);
        }
        res->create_kernel();
        return std::make_shared<const JitQsaScore>(std::move(res));
    };
    try {
        return cache->getOrCreate(key, builder).first;
    } catch (const ov::Exception&) {
        return nullptr;
    }
}

JitQsaScore::JitQsaScore(std::shared_ptr<kernel::JitKernelBase> kernel) : m_kernel(std::move(kernel)) {}

void JitQsaScore::operator()(const float* q, const void* rows, size_t n, float* scores, float* stage) const {
    kernel::jit_qsa_score_call_args args{q, rows, scores, stage, n};
    (*m_kernel)(&args);
}

}  // namespace ov::intel_cpu::node::qsa
