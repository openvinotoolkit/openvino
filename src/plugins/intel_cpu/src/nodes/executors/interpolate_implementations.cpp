// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common/memory_desc_wrapper.hpp>
#include <memory>
#include <optional>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_types.h"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mlas/mlas_gemm.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

#include "nodes/executors/common/ref_interpolate.hpp"


namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

static const MappingNotation refInterpolateMappingNotation{ARG_SRC, ARG_DST};

using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig refInterpolateLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

//template <dnnl::impl::cpu::x64::cpu_isa_t ISA>
//struct Require {
//    bool operator()() {
//        return dnnl::impl::cpu::x64::mayiuse(ISA);
//    }
//};

// clang-format off
static const TypeMapping refInterpolateTypeMapping {
    {{_any, _any}, pt(just<f32>(), just<f32>())},
};

template <>
const std::vector<ExecutorImplementation<InterpolateAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<InterpolateAttrs>> interpolateImplementations {
        OV_CPU_INSTANCE_COMMON(
            "interpolate_ref",
            ExecutorType::Common,
            OperationType::Interpolate,
            ShapeTolerance::Agnostic,
            // supports
            [](const InterpolateConfig& config) -> bool {
                return CommonInterpolateExecutor::supports(config);
            },
            // requiresFallback
            [](const InterpolateConfig& config) -> std::optional<executor::Config<InterpolateAttrs>> {
                return requiresFallbackCommon(config,
                                              refInterpolateTypeMapping,
                                              refInterpolateLayoutConfig,
                                              refInterpolateMappingNotation);
            },
            AcceptsAnyShape<InterpolateAttrs>{},
            CreateDefault<CommonInterpolateExecutor, InterpolateAttrs>{}
            )
    };
    return interpolateImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu