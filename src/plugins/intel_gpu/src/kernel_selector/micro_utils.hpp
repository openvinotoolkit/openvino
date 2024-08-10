// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_ONEDNN_FOR_GPU

#ifdef UNUSED
#   undef UNUSED
#endif

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include "gpu/intel/microkernels/package.hpp"
#include "gpu/intel/jit/gemm/microkernel_provider.hpp"
#include "gpu/intel/jit/gemm/gen_gemm_kernel_generator.hpp"
#include "gpu/intel/microkernels/shim.hpp"

namespace micro {

using Package = dnnl::impl::gpu::intel::micro::Package;
using HWInformation = dnnl::impl::gpu::intel::jit::HWInformation;
using GEMMProblem = dnnl::impl::gpu::intel::jit::GEMMProblem;
using GEMMStrategy = dnnl::impl::gpu::intel::jit::GEMMStrategy;
using GEMMProtocol = dnnl::impl::gpu::intel::micro::GEMMProtocol;
using MatrixLayout = dnnl::impl::gpu::intel::jit::MatrixLayout;
using Type = dnnl::impl::gpu::intel::jit::Type;
using SizeParams = dnnl::impl::gpu::intel::jit::SizeParams;
using StrategyRequirement = dnnl::impl::gpu::intel::jit::StrategyRequirement;
using ShimOptions = dnnl::impl::gpu::intel::micro::ShimOptions;
using HostLanguage = dnnl::impl::gpu::intel::micro::HostLanguage;

// Wrapper for Package which is used in clKernelData with forward declaration
// to avoid including this header in many places in plugin
// which may cause symbols conflicts with oneDNN
struct MicroKernelPackage {
    explicit MicroKernelPackage(Package _p) : p(_p) {}
    Package p;
};

inline Package select_gemm_microkernel(GEMMProtocol protocol, HWInformation hw_info, SizeParams sizes, const GEMMProblem &problem,
                                        const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                                        void (*strategyAdjuster)(GEMMStrategy &strategy) = nullptr) {
    return dnnl::impl::gpu::intel::jit::selectGEMMMicrokernel(protocol, hw_info, sizes, problem, reqs, strategyAdjuster);
}

static inline int alignment_for_ld(int ld) {
    return  dnnl::impl::gpu::intel::jit::alignmentForLD(ld);
}

}  // namespace micro

#undef UNUSED

#endif  // ENABLE_ONEDNN_FOR_GPU
