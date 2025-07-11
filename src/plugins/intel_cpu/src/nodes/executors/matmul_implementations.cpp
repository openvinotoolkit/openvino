// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/general_utils.h"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/executors/x64/matmul_small.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_fullyconnected.hpp"
#    include "nodes/executors/common/common_utils.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig dnnlMatMulLayoutConfig{LayoutType::ncsp,
                                                 LayoutType::ncsp,
                                                 LayoutType::ncsp,
                                                 LayoutType::ncsp};

template <dnnl::impl::cpu::x64::cpu_isa_t ISA>
struct Require {
    bool operator()() {
        return dnnl::impl::cpu::x64::mayiuse(ISA);
    }
};

// clang-format off
static const TypeMapping dnnlMatMulTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},               {bypass(), bypass(), use<3>(), bypass()}},
    {{_f16, _f16, _any, _f16 | _f32},                         {bypass(), bypass(), use<3>(), bypass()}},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            {bypass(), bypass(), use<0>(), use<0>()}},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any},                  {bypass(), bypass(), use<0>(), use<0>()}},
    {{_bf16, _f16, _any, _any},                        {bypass(), bypass(), use<0>(), use<0>()}},
    {{_f16, _bf16, _any, _any},                        {bypass(), bypass(), use<0>(), use<0>()}},
    // quantization configuration
    {{_u8 | _i8, _i8, _u8|_i8|_i32|_bf16|_f16|_f32|_dynamic, _u8|_i8|_i32|_bf16|_f16|_f32}, {bypass(), bypass(), bypass(),  bypass()}},
    {{_u8 | _i8, _i8, _any, _any},                            {bypass(), bypass(), just<f32>(), just<f32>()}},
    // compresses int weights
    {{_f32 | _bf16 | _f16, _u8 | _i8, _any, _any},            {bypass(), bypass(), use<0>(), use<0>()}},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                {just<f32>(), just<f32>(), just<f32>(), just<f32>()}},
};

#if defined(OV_CPU_WITH_ACL)
static const TypeMapping aclMatMulTypeMapping {
    // {src, wei, bia, dst}                  pt<src, wei, bias, dst>
    {{_f32 | _f16, _f32 | _f16, _any, _any}, {bypass(), bypass(), use<0>(), use<0>()}},
    {{_any, _any, _any, _any},               {just<f32>(), just<f32>(), just<f32>(), just<f32>()}}}
};
#endif

static const MappingNotation matmulMappingNotation {
    {ARG_SRC,  0},
    {ARG_WEI,  1},
    {ARG_BIAS, 2},
    {ARG_DST,  3}
};
// clang-format on

[[maybe_unused]] static inline bool noWeightsDecompression(const MatMulConfig& config) {
    return !DnnlMatMulPrimitive::useWeightsDecompressionImpl(srcType(config), weiType(config));
}

[[maybe_unused]] static inline bool noSparseDecompression(const MatMulConfig& config) {
    return !(config.attrs.sparseWeights);
}

[[maybe_unused]] static inline bool noPostOps(const MatMulConfig& config) {
    return config.attrs.postOps.empty();
}

struct CreateOptimalConfigDefault {
    std::optional<MatMulConfig> operator()(const MatMulConfig& config) const {
        return createOptimalConfigCommon(config, dnnlMatMulTypeMapping, dnnlMatMulLayoutConfig, matmulMappingNotation);
    }
};

// clang-format off
template <>
const std::vector<ExecutorImplementation<MatMulAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<MatMulAttrs>> matmulImplementations {
        OV_CPU_INSTANCE_X64(
            "matmul_small_x64",
            ExecutorType::Jit,
            OperationType::MatMul,
            // supports
            [](const MatMulConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                VERIFY(all_of(f32, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);
                
                const auto& srcDesc = config.descs.at(ARG_SRC);
                const auto& weiDesc = config.descs.at(ARG_WEI);
                const auto srcRank = srcDesc->getShape().getRank();
                const auto weiRank = weiDesc->getShape().getRank();
                
                return srcRank >= 2 && srcRank == weiRank;
            },
            HasNoOptimalConfig<MatMulAttrs>{},
            []([[maybe_unused]] const MatMulAttrs& attrs, const MemoryArgs& memory) -> bool {
                const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
                const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
                const auto& srcShape = srcDesc->getShape().getStaticDims();
                const auto& weiShape = weiDesc->getShape().getStaticDims();
                const auto srcRank = srcShape.size();
                const auto weiRank = weiShape.size();
                
                for (size_t i = 0; i < srcRank - 2; i++) {
                    if (srcShape[i] != weiShape[i]) {
                        return false;
                    }
                }
                
                return (srcShape[srcRank - 1] <= 2) && (srcShape[srcRank - 2] <= 2) && 
                       (weiShape[weiRank - 1] <= 2) && (weiShape[weiRank - 2] <= 2);
            },
            CreateDefault<MatMulSmallExecutor, MatMulAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL(
            "matmul_dnnl",
            ExecutorType::Dnnl,
            OperationType::MatMul,
            SupportsAnyConfig<MatMulAttrs>{},
            // createOptimalConfig
            [](const MatMulConfig& config) -> std::optional<executor::Config<MatMulAttrs>> {
                return createOptimalConfigCommon(config,
                                                 dnnlMatMulTypeMapping,
                                                 dnnlMatMulLayoutConfig,
                                                 matmulMappingNotation);
            },
            AcceptsAnyShape<MatMulAttrs>,
            CreateDnnlDefault<DnnlMatMulPrimitive, MatMulAttrs>{true, true}
            )
    };

    return matmulImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
