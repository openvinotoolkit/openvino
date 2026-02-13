// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/common/shape_inference.hpp"

#include "snippets/shape_inference/shape_infer_instances.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"

namespace ov::snippets {
const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry = []() {
    auto registry = detail::make_common_cpu_shape_infer_registry();
    registry.insert(make_specific_external<ov::intel_cpu::aarch64::GemmCPU, BrgemmShapeInfer>());
    registry.insert(make_specific<ov::intel_cpu::aarch64::GemmCopyB>());
    return registry;
}();

}  // namespace ov::snippets
