// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference.hpp"

#include "snippets/shape_inference/shape_infer_instances.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
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

#define SHAPE_INFER_PREDEFINED(OP, InferType)                                \
    {                                                                        \
        OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { \
            return std::make_shared<InferType>();                            \
        }                                                                    \
    }
#define SHAPE_INFER_OP_SPECIFIC(OP)                                          \
    {                                                                        \
        OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { \
            return std::make_shared<OP::ShapeInfer>(n);                      \
        }                                                                    \
    }
#define SHAPE_INFER_OP_SPECIFIC_EXTERNAL(OP, InferType)                      \
    {                                                                        \
        OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { \
            return std::make_shared<InferType>(n);                           \
        }                                                                    \
    }

const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry{
    SHAPE_INFER_PREDEFINED(ov::intel_cpu::FusedMulAdd, NumpyBroadcastShapeInfer),
    SHAPE_INFER_PREDEFINED(ov::intel_cpu::SwishNode, PassThroughShapeInfer),
    SHAPE_INFER_OP_SPECIFIC_EXTERNAL(ov::intel_cpu::tpp::op::BrgemmTPP, BrgemmShapeInfer),
};
#undef SHAPE_INFER_OP_SPECIFIC
#undef SHAPE_INFER_PREDEFINED

}  // namespace ov::snippets
