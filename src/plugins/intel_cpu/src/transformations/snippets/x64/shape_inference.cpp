// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference.hpp"
#include <snippets/lowered/shape_inference/shape_infer_instances.hpp>
#include "op/brgemm_copy_b.hpp"
#include "op/brgemm_cpu.hpp"
#include "op/fused_mul_add.hpp"
#include "op/load_convert.hpp"
#include "op/store_convert.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"

namespace ov {
namespace snippets {
using ShapeInferPtr = IShapeInferSnippetsFactory::ShapeInferPtr;

ShapeInferPtr CPUShapeInferSnippetsFactory::get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key,
                                                                        const std::shared_ptr<ov::Node>& op) const {
    const auto& maker_iter = specific_ops_registry.find(key);
    if (maker_iter != specific_ops_registry.end()) {
        return maker_iter->second(op);
    } else {
        return {};
    }
}


#define SHAPE_INFER_PREDEFINED(OP, InferType) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<InferType>();} }
#define SHAPE_INFER_OP_SPECIFIC(OP) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<OP::ShapeInfer>(n);} }

const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry {
        // todo: Parameter and Scalar should be handled separately, since they have no inputs, and infer can't be called
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::FusedMulAdd, entryNumpyBroadcasting),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::SwishNode, entryFirstPassthrough),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::LoadConvertSaturation, entryFirstPassthrough),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::LoadConvertTruncation, entryFirstPassthrough),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::StoreConvertSaturation, entryFirstPassthrough),
        SHAPE_INFER_PREDEFINED(ov::intel_cpu::StoreConvertTruncation, entryFirstPassthrough),
        //
        SHAPE_INFER_OP_SPECIFIC(ov::intel_cpu::BrgemmCopyB),
        SHAPE_INFER_OP_SPECIFIC(ov::intel_cpu::BrgemmCPU),
};
#undef SHAPE_INFER_OP_SPECIFIC
#undef SHAPE_INFER_PREDEFINED

} // namespace snippets
} // namespace ov
