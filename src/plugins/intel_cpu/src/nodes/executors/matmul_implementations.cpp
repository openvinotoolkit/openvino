// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"

#ifdef OPENVINO_ARCH_X86_64
#    include <cstddef>

#    include "debug_messages.hpp"
#    include "nodes/executors/x64/matmul_small.hpp"
#    include "utils/general_utils.h"
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
    {{_any, _any, _any, _any},               {just<f32>(), just<f32>(), just<f32>(), just<f32>()}}
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

                VERIFY(!config.attrs.transposeA && !config.attrs.transposeB, "unsupported strides");
                VERIFY(all_of(f32, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);

                const auto& biasDesc = config.descs.at(ARG_BIAS);
                VERIFY(biasDesc->empty(), "bias is not supported");
                
                const auto& srcDesc0 = config.descs.at(ARG_SRC);
                const auto& srcDesc1 = config.descs.at(ARG_WEI);
                const auto srcRank0 = srcDesc0->getShape().getRank();
                const auto srcRank1 = srcDesc1->getShape().getRank();

                VERIFY(srcRank0 >= 2, UNSUPPORTED_SRC_RANK);
                VERIFY(srcRank0 == srcRank1, UNSUPPORTED_SRC_RANK);

                return true;
            },
            HasNoOptimalConfig<MatMulAttrs>{},
            []([[maybe_unused]] const MatMulAttrs& attrs, const MemoryArgs& memory) -> bool {
                const auto& srcDesc0 = memory.at(ARG_SRC)->getDescPtr();
                const auto& srcDesc1 = memory.at(ARG_WEI)->getDescPtr();
                const auto& srcShape0 = srcDesc0->getShape().getStaticDims();
                const auto& srcShape1 = srcDesc1->getShape().getStaticDims();
                const auto srcRank0 = srcShape0.size();
                const auto srcRank1 = srcShape1.size();
                
                for (size_t i = 0; i < srcRank0 - 2; i++) {
                    if (srcShape0[i] != srcShape1[i]) {
                        return false;
                    }
                }

                return (srcShape0[srcRank0 - 1] <= 2) && (srcShape0[srcRank0 - 2] <= 2) && 
                       (srcShape1[srcRank1 - 1] <= 2) && (srcShape1[srcRank1 - 2] <= 2);
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
