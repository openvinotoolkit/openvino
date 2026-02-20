// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/common/shape_inference.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/visibility.hpp"
#include "snippets/shape_inference/shape_infer_instances.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"

#ifdef SNIPPETS_LIBXSMM_TPP
#    include "transformations/tpp/common/op/brgemm.hpp"
#endif

namespace ov::snippets {
using ShapeInferPtr = IShapeInferSnippetsFactory::ShapeInferPtr;

namespace detail {

IShapeInferSnippetsFactory::TRegistry make_common_cpu_shape_infer_registry() {
    const static IShapeInferSnippetsFactory::TRegistry registry{
        {ov::intel_cpu::FusedMulAdd::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>&) {
             return std::make_shared<NumpyBroadcastShapeInfer>();
         }},
        {ov::intel_cpu::SwishNode::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>&) {
             return std::make_shared<PassThroughShapeInfer>();
         }},
#ifdef SNIPPETS_LIBXSMM_TPP
        {ov::intel_cpu::tpp::op::BrgemmTPP::get_type_info_static(),
         [](const std::shared_ptr<ov::Node>& n) {
             return std::make_shared<BrgemmShapeInfer>(n);
         }},
#endif
    };
    return registry;
}

}  // namespace detail

ShapeInferPtr CPUShapeInferSnippetsFactory::get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key,
                                                                        const std::shared_ptr<ov::Node>& op) const {
    const auto& maker_iter = specific_ops_registry.find(key);
    if (maker_iter != specific_ops_registry.end()) {
        return maker_iter->second(op);
    }
    return {};
}

#if !defined(OPENVINO_ARCH_X86_64) && !defined(OPENVINO_ARCH_ARM64)
const CPUShapeInferSnippetsFactory::TRegistry CPUShapeInferSnippetsFactory::specific_ops_registry =
    detail::make_common_cpu_shape_infer_registry();
#endif

}  // namespace ov::snippets
