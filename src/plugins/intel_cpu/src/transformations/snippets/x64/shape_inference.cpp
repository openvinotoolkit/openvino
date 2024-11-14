// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference.hpp"
#include <snippets/shape_inference/shape_infer_instances.hpp>
#include "op/brgemm_copy_b.hpp"
#include "op/brgemm_cpu.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "op/load_convert.hpp"
#include "op/store_convert.hpp"
#include "op/perf_count_rdtsc.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#ifdef SNIPPETS_LIBXSMM_TPP
#include "transformations/tpp/x64/op/brgemm.hpp"
#include "transformations/tpp/x64/op/equation.hpp"
#include "transformations/tpp/x64/op/scalar.hpp"
#include "transformations/tpp/x64/op/reduce.hpp"
#endif

namespace ov {
namespace snippets {
using ShapeInferPtr = IShapeInferSnippetsFactory::ShapeInferPtr;

ShapeInferPtr CPUShapeInferSnippetsFactory::get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key,
                                                                        const std::shared_ptr<ov::Node>& op) const {
    const auto& maker_iter = specific_ops_registry.find(key);
    if (maker_iter != specific_ops_registry.end())
        return maker_iter->second(op);
    return {};
}


#define SHAPE_INFER_PREDEFINED(OP, InferType) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<InferType>();} }
#define SHAPE_INFER_OP_SPECIFIC(OP) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<OP::ShapeInfer>(n);} }
#define SHAPE_INFER_OP_SPECIFIC_EXTERNAL(OP, InferType) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<InferType>(n);} }

const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry {
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::FusedMulAdd, NumpyBroadcastShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::SwishNode, PassThroughShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::LoadConvertSaturation, PassThroughShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::LoadConvertTruncation, PassThroughShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::StoreConvertSaturation, PassThroughShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::StoreConvertTruncation, PassThroughShapeInfer),
#ifdef SNIPPETS_DEBUG_CAPS
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::PerfCountRdtscBegin, EmptyShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::PerfCountRdtscEnd, EmptyShapeInfer),
#endif
#ifdef SNIPPETS_LIBXSMM_TPP
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(ov::intel_cpu::tpp::op::BrgemmTPP, BrgemmShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::tpp::op::EquationTPP, NumpyBroadcastShapeInfer),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::tpp::op::Scalar, SingleElementShapeInfer),
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(ov::intel_cpu::tpp::op::ReduceMax, ReduceShapeInfer),
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(ov::intel_cpu::tpp::op::ReduceSum, ReduceShapeInfer),
#endif
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(ov::intel_cpu::BrgemmCPU, BrgemmShapeInfer),
        SHAPE_INFER_OP_SPECIFIC(ov::intel_cpu::BrgemmCopyB),
};
#undef SHAPE_INFER_OP_SPECIFIC
#undef SHAPE_INFER_PREDEFINED

} // namespace snippets
} // namespace ov
