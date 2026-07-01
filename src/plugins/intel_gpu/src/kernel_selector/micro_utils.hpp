// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_ONEDNN_FOR_GPU

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"

#ifdef UNUSED
#   undef UNUSED
#endif

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/package.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/kernel_selector.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel_selector.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/microkernel/shim.hpp"
#include "common/utils.hpp"

namespace micro {

using Package = gemmstone::microkernel::Package;
using HWInformation = gemmstone::microkernel::HWInformation;
using GEMMProblem = gemmstone::GEMMProblem;
using ABOffset = gemmstone::ABOffset;
using GEMMStrategy = gemmstone::GEMMStrategy;
using GEMMProtocol = gemmstone::microkernel::Protocol;
using GEMMOptions = gemmstone::microkernel::GEMMOptions;
using MatrixLayout = gemmstone::MatrixLayout;
using Type = gemmstone::Type;
using SizeParams = gemmstone::SizeParams;
using StrategyRequirement = gemmstone::StrategyRequirement;
using ShimOptions = gemmstone::microkernel::ShimOptions;
using HostLanguage = gemmstone::microkernel::HostLanguage;
using Setting = gemmstone::microkernel::Package::Setting;

using dnnl::impl::utils::rnd_up_pow2;

// Wrapper for Package which is used in clKernelData with forward declaration
// to avoid including this header in many places in plugin
// which may cause symbols conflicts with oneDNN
struct MicroKernelPackage {
    MicroKernelPackage() = default;
    explicit MicroKernelPackage(Package _p) : p(_p) {}
    Package p;

    // WARNING: We serialize only microkernels settings, so after deserialization
    // other struct fields are not initializer properly and can't be used
    void save(cldnn::BinaryOutputBuffer& ob) const {
        ob << p.settings.size();
        for (auto& s : p.settings) {
            ob << s.name;
            ob << s.value;
        }
    }

    void load(cldnn::BinaryInputBuffer& ib) {
        size_t n_settings;
        ib >> n_settings;
        p.settings.clear();
        for (size_t i = 0; i < n_settings; i++) {
            Setting s;
            ib >> s.name;
            ib >> s.value;
            p.settings.push_back(s);
        }
    }
};

inline Package select_gemm_microkernel(GEMMOptions &options, HWInformation hw_info, SizeParams sizes, const GEMMProblem &problem,
                                        const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                                        void (*strategyAdjuster)(GEMMStrategy &strategy) = nullptr, gemmstone::SelectionObserver *observer = nullptr) {
    return gemmstone::microkernel::selectGEMM(options, hw_info, sizes, problem, reqs, strategyAdjuster);
}
inline Package select_gemm_microkernel(GEMMOptions &options, HWInformation hw_info, SizeParams sizes, const GEMMProblem &problem,
        gemmstone::SelectionObserver *observer) {
    return gemmstone::microkernel::selectGEMM(options, hw_info, sizes, problem, {}, nullptr);
}

static inline int alignment_for_ld(int ld) {
    return  gemmstone::microkernel::alignmentForLD(ld);
}

}  // namespace micro

#undef UNUSED

#endif  // ENABLE_ONEDNN_FOR_GPU
