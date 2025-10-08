// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/common/interpolate_common_executor.hpp"
#include "nodes/executors/jit/interpolate_jit_wrapper.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_interpolate.hpp"
#endif
#include "utils/arch_macros.h"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov::intel_cpu {

using namespace executor;
using namespace TypeMaskAlias;

static const TypeMapping interpolateTypeMapping{
    // {src, dst}         pt<src, dst>
    {{_any, _any},        {bypass(), bypass()}},
};

static const MappingNotation interpolateMappingNotation{{ARG_SRC, 0}, {ARG_DST, 0}};

struct createOptimalConfigDefault {
    std::optional<InterpolateConfig> operator()(const InterpolateConfig& config) const {
        return createOptimalConfigCommon(config, interpolateTypeMapping, layoutConfig, interpolateMappingNotation);
    }

    std::vector<LayoutType> layoutConfig;
};

using CreateCommon = CreateDefault<InterpolateCommonExecutor, InterpolateAttrs>;
using CreateJit = CreateDefault<InterpolateJitExecutorWrapper, InterpolateAttrs>;

template <>
const std::vector<ExecutorImplementation<InterpolateAttrs>>& getImplementations<InterpolateAttrs>() {
    static const std::vector<ExecutorImplementation<InterpolateAttrs>> impls{
        OV_CPU_INSTANCE_X64(
            "interpolate_jit_nspc",
            ExecutorType::Jit,
            OperationType::Interpolate,
            [](const InterpolateConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config.descs, {LayoutType::nspc}, memoryFormatFilter, interpolateMappingNotation)) return false;
                // Support only 4D/5D tensors for JIT
                const auto& srcDesc = config.descs.at(ARG_SRC);
                const auto rank = srcDesc->getShape().getRank();
                if (rank != 4 && rank != 5) return false;
                // JIT path does not implement InterpolateMode::linear
                if (config.attrs.mode == ov::intel_cpu::InterpolateMode::linear) return false;
                // NCHWAsNHWC handled by ref path
                if (config.attrs.NCHWAsNHWC) return false;
                return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
            },
            createOptimalConfigDefault{{LayoutType::nspc}},
            AcceptsAnyShape<InterpolateAttrs>,
            CreateJit{})
        OV_CPU_INSTANCE_X64(
            "interpolate_jit_ncsp",
            ExecutorType::Jit,
            OperationType::Interpolate,
            [](const InterpolateConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config.descs, {LayoutType::ncsp}, memoryFormatFilter, interpolateMappingNotation)) return false;
                const auto& srcDesc = config.descs.at(ARG_SRC);
                const auto rank = srcDesc->getShape().getRank();
                if (rank != 4 && rank != 5) return false;
                if (config.attrs.mode == ov::intel_cpu::InterpolateMode::linear) return false;
                if (config.attrs.NCHWAsNHWC) return false;
                return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
            },
            createOptimalConfigDefault{{LayoutType::ncsp}},
            AcceptsAnyShape<InterpolateAttrs>,
            CreateJit{})
        OV_CPU_INSTANCE_X64(
            "interpolate_jit_nCsp8c",
            ExecutorType::Jit,
            OperationType::Interpolate,
            [](const InterpolateConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config.descs, {LayoutType::nCsp8c}, memoryFormatFilter, interpolateMappingNotation)) return false;
                const auto& srcDesc = config.descs.at(ARG_SRC);
                const auto rank = srcDesc->getShape().getRank();
                if (rank != 4 && rank != 5) return false;
                if (config.attrs.mode == ov::intel_cpu::InterpolateMode::linear) return false;
                if (config.attrs.NCHWAsNHWC) return false;
                return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
            },
            createOptimalConfigDefault{{LayoutType::nCsp8c}},
            AcceptsAnyShape<InterpolateAttrs>,
            CreateJit{})
        OV_CPU_INSTANCE_X64(
            "interpolate_jit_nCsp16c",
            ExecutorType::Jit,
            OperationType::Interpolate,
            [](const InterpolateConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config.descs, {LayoutType::nCsp16c}, memoryFormatFilter, interpolateMappingNotation)) return false;
                const auto& srcDesc = config.descs.at(ARG_SRC);
                const auto rank = srcDesc->getShape().getRank();
                if (rank != 4 && rank != 5) return false;
                if (config.attrs.mode == ov::intel_cpu::InterpolateMode::linear) return false;
                if (config.attrs.NCHWAsNHWC) return false;
                return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
            },
            createOptimalConfigDefault{{LayoutType::nCsp16c}},
            AcceptsAnyShape<InterpolateAttrs>,
            CreateJit{})
        OV_CPU_INSTANCE_ACL(
            "interpolate_acl_nspc",
            ExecutorType::Acl,
            OperationType::Interpolate,
            [](const InterpolateConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config.descs, {LayoutType::nspc}, memoryFormatFilter, interpolateMappingNotation)) return false;
                return AclInterpolateExecutor::supports(config);
            },
            createOptimalConfigDefault{{LayoutType::nspc}},
            AcceptsAnyShape<InterpolateAttrs>,
            CreateDefault<AclInterpolateExecutor, InterpolateAttrs>{})
        OV_CPU_INSTANCE_COMMON(
            "interpolate_common_ncsp",
            ExecutorType::Common,
            OperationType::Interpolate,
            [](const InterpolateConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                return MatchesMemoryFormatFilter(config.descs, {LayoutType::ncsp}, memoryFormatFilter, interpolateMappingNotation);
            },
            createOptimalConfigDefault{{LayoutType::ncsp}},
            AcceptsAnyShape<InterpolateAttrs>,
            CreateCommon{})
        OV_CPU_INSTANCE_COMMON(
            "interpolate_common_nspc",
            ExecutorType::Common,
            OperationType::Interpolate,
            [](const InterpolateConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                return MatchesMemoryFormatFilter(config.descs, {LayoutType::nspc}, memoryFormatFilter, interpolateMappingNotation);
            },
            createOptimalConfigDefault{{LayoutType::nspc}},
            AcceptsAnyShape<InterpolateAttrs>,
            CreateCommon{})
    };

    return impls;
}

}  // namespace ov::intel_cpu
