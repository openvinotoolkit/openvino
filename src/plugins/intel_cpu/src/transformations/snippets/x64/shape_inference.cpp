// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/common/shape_inference.hpp"

#include <snippets/shape_inference/shape_infer_instances.hpp>

#include "op/brgemm_copy_b.hpp"
#include "op/brgemm_cpu.hpp"
#include "op/load_convert.hpp"
#include "op/perf_count_rdtsc.hpp"
#include "op/store_convert.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov::snippets {
const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry = []() {
    auto registry = detail::make_common_cpu_shape_infer_registry();

    registry.insert(make_predefined<ov::intel_cpu::LoadConvertSaturation, PassThroughShapeInfer>());
    registry.insert(make_predefined<ov::intel_cpu::LoadConvertTruncation, PassThroughShapeInfer>());
    registry.insert(make_predefined<ov::intel_cpu::StoreConvertSaturation, PassThroughShapeInfer>());
    registry.insert(make_predefined<ov::intel_cpu::StoreConvertTruncation, PassThroughShapeInfer>());
#ifdef SNIPPETS_DEBUG_CAPS
    registry.insert(make_predefined<ov::intel_cpu::PerfCountRdtscBegin, EmptyShapeInfer>());
    registry.insert(make_predefined<ov::intel_cpu::PerfCountRdtscEnd, EmptyShapeInfer>());
#endif
    registry.insert(make_specific_external<ov::intel_cpu::BrgemmCPU, BrgemmShapeInfer>());
    registry.insert(make_specific<ov::intel_cpu::BrgemmCopyB>());

    return registry;
}();

}  // namespace ov::snippets
