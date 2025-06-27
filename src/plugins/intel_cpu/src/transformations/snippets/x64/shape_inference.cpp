// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/common/shape_inference.hpp"

#include <memory>
#include <snippets/shape_inference/shape_infer_instances.hpp>

#include "op/brgemm_copy_b.hpp"
#include "op/brgemm_cpu.hpp"
#include "op/load_convert.hpp"
#include "op/perf_count_rdtsc.hpp"
#include "op/store_convert.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#ifdef SNIPPETS_LIBXSMM_TPP
#    include "transformations/tpp/common/op/brgemm.hpp"
#    include "transformations/tpp/x64/op/equation.hpp"
#    include "transformations/tpp/x64/op/reduce.hpp"
#    include "transformations/tpp/x64/op/scalar.hpp"
#endif

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
    make_predefined<ov::intel_cpu::LoadConvertSaturation, PassThroughShapeInfer>(),
    make_predefined<ov::intel_cpu::LoadConvertTruncation, PassThroughShapeInfer>(),
    make_predefined<ov::intel_cpu::StoreConvertSaturation, PassThroughShapeInfer>(),
    make_predefined<ov::intel_cpu::StoreConvertTruncation, PassThroughShapeInfer>(),
#ifdef SNIPPETS_DEBUG_CAPS
    make_predefined<ov::intel_cpu::PerfCountRdtscBegin, EmptyShapeInfer>(),
    make_predefined<ov::intel_cpu::PerfCountRdtscEnd, EmptyShapeInfer>(),
#endif
#ifdef SNIPPETS_LIBXSMM_TPP
    make_specific_external<ov::intel_cpu::tpp::op::BrgemmTPP, BrgemmShapeInfer>(),
    make_predefined<ov::intel_cpu::tpp::op::EquationTPP, NumpyBroadcastShapeInfer>(),
    make_predefined<ov::intel_cpu::tpp::op::Scalar, SingleElementShapeInfer>(),
    make_specific_external<ov::intel_cpu::tpp::op::ReduceMax, ReduceShapeInfer>(),
    make_specific_external<ov::intel_cpu::tpp::op::ReduceSum, ReduceShapeInfer>(),
#endif
    make_specific_external<ov::intel_cpu::BrgemmCPU, BrgemmShapeInfer>(),
    make_specific<ov::intel_cpu::BrgemmCopyB>(),
};

}  // namespace ov::snippets
