// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/common/shape_inference.hpp"

#include "snippets/shape_inference/shape_infer_instances.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "transformations/snippets/riscv64/op/brgemm_cpu.hpp"

namespace ov::snippets {

const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry = []() {
    auto registry = detail::make_common_cpu_shape_infer_registry();
    registry.insert(make_specific_external<ov::intel_cpu::BrgemmCPU, BrgemmShapeInfer>());
    return registry;
}();

}  // namespace ov::snippets
