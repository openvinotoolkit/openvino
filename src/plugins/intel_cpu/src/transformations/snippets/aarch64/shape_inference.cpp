// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/common/shape_inference.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "snippets/shape_inference/shape_infer_instances.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

namespace ov::snippets {
using ShapeInferPtr = IShapeInferSnippetsFactory::ShapeInferPtr;

ShapeInferPtr CPUShapeInferSnippetsFactory::get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key,
                                                                        const std::shared_ptr<ov::Node>& op) const {
    const auto& maker_iter = specific_ops_registry.find(key);
    if (maker_iter != specific_ops_registry.end()) {
        return maker_iter->second(op);
    }
    return {};
}

const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry{
    make_predefined<ov::intel_cpu::FusedMulAdd, NumpyBroadcastShapeInfer>(),
    make_predefined<ov::intel_cpu::SwishNode, PassThroughShapeInfer>(),
    make_specific_external<ov::intel_cpu::tpp::op::BrgemmTPP, BrgemmShapeInfer>(),
    make_specific_external<ov::intel_cpu::aarch64::GemmCPU, BrgemmShapeInfer>(),
    make_specific<ov::intel_cpu::aarch64::GemmCopyB>(),
};

}  // namespace ov::snippets
