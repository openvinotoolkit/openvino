// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/gemm_config.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "ov_optional.hpp"
#include "utils/cpp/maybe_unused.hpp"

#if defined(OV_CPU_WITH_ACL)
#include "nodes/executors/acl/acl_fullyconnected.hpp"
#include "nodes/executors/acl/acl_gemm.hpp"
#endif

namespace ov {
namespace intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

static const MappingNotation dnnlFCMappingNotation{ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST};

using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig dnnlFCLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};
static const LayoutConfig aclFCLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};
static const LayoutConfig aclMatMulLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

template<dnnl::impl::cpu::x64::cpu_isa_t ISA>
struct Require {
    bool operator()() {
        return dnnl::impl::cpu::x64::mayiuse(ISA);
    }
};

static const TypeMapping aclMatMulTypeMapping {
    // {src, wei, bia, dst}              pt<src, wei, bias, dst>
    {{_i8, _i8, _any, _any},          pt(just<i8>(), just<i8>(), just<i32>(), just<i32>())},
    {{_u8, _u8, _any, _any},          pt(just<u8>(), just<u8>(), just<i32>(), just<i32>())},
    {{_any, _any, _any, _any},        pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())}
};

static const MappingNotation aclMatMulMappingNotation {
    ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST
};

// clang-format on

static bool fullyMatchConfiguration(const MemoryDescArgs& currentDescriptors,
                                    const InOutTypes& typeConfig,
                                    const LayoutConfig& layoutConfig,
                                    const MappingNotation& notation) {
    for (size_t i = 0; i < typeConfig.size(); i++) {
        const auto& type = typeConfig[i];
        const auto& desc = currentDescriptors.at(notation[i]);

        if (desc->empty())
            continue;

        if (desc->getPrecision() != type)
            return false; // type mismatch

        if (!desc->hasLayoutType(layoutConfig[i]))
            return false; // layout mismatch
    }

    return true;
}

static MemoryDescArgs createOptimalDescriptors(const MemoryDescArgs& currentDescriptors,
                                               const InOutTypes& typeConfig,
                                               const LayoutConfig& layoutConfig,
                                               const MappingNotation& notation) {
    MemoryDescArgs descs = currentDescriptors;

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < typeConfig.size(); i++) {
        const auto& desc = currentDescriptors.at(notation[i]);
        const auto& descType = desc->getPrecision();
        const auto& type = typeConfig[i];
        const auto& layout = layoutConfig[i];

        if (desc->empty())
            continue;

        if (descType == type && desc->hasLayoutType(layout)) {
            continue;
        }

        descs[notation[i]] = creatorsMap.at(layout)->createSharedDesc(type, desc->getShape());
    }

    return descs;
}

template <typename Attrs>
ov::optional<executor::Config<Attrs>> requiresFallbackCommon(const executor::Config<Attrs>& config,
                                                             const TypeMapping& typeMapping,
                                                             const LayoutConfig& layoutConfig,
                                                             const MappingNotation& notation) {
    const auto typeConfig = getTypeConfiguration(config.descs, typeMapping, notation);

    if (fullyMatchConfiguration(config.descs, typeConfig, layoutConfig, notation)) {
        return {};
    }

    const auto optimalDescriptors = createOptimalDescriptors(config.descs, typeConfig, layoutConfig, notation);

    return ov::optional<executor::Config<Attrs>>(GEMMConfig{optimalDescriptors, config.attrs, config.postOps});
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noWeightsDecompression(const GEMMConfig& config) {
    return !DnnlFCPrimitive::useWeightsDecompressionImpl(srcType(config), weiType(config), config.attrs.modelType);
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noSparseDecompression(const GEMMConfig& config) {
    return !(config.attrs.sparseWeights);
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noPostOps(const FCConfig& config) {
    return config.postOps.empty();
}

template <>
const std::vector<ExecutorImplementation<GEMMAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<GEMMAttrs>> gemmImplementations {
        OV_CPU_INSTANCE_ACL(
            "matmul_acl",
            ExecutorType::Acl,
            OperationType::MatMul,
            ShapeTolerance::Agnostic,
            // supports
            [](const GEMMConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                return ACLGEMMExecutor::supports(config);
            },
            // requiresFallback
            [](const GEMMConfig& config) -> ov::optional<executor::Config<GEMMAttrs>> {
                return requiresFallbackCommon(config,
                                              aclMatMulTypeMapping,
                                              aclMatMulLayoutConfig,
                                              aclMatMulMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const GEMMAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<ACLGEMMExecutor>(attrs, postOps, memory, context);
            })
    };

    return gemmImplementations;
}
}  // namespace intel_cpu
}  // namespace ov
