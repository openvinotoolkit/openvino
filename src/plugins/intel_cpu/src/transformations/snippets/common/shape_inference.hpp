// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <snippets/shape_inference/shape_inference.hpp>

#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"

namespace ov::snippets {

/**
 * \brief Shape infer factory that can create shape-infer instances for cpu-specific operations
 */
class CPUShapeInferSnippetsFactory : public IShapeInferSnippetsFactory {
    /** \brief Factory makers registry which can be specialized for key and value. */
    static const TRegistry specific_ops_registry;

protected:
    /**
     * @brief get shape infer instances for operations from backend-specific opset
     * @return register ShapeInferPtr
     */
    [[nodiscard]] ShapeInferPtr get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key,
                                                            const std::shared_ptr<ov::Node>& op) const override;

    // Helper to register an operation with a stateless shape infer implementation
    template <typename Op, typename InferType>
    static IShapeInferSnippetsFactory::TRegistry::value_type make_predefined() {
        return {Op::get_type_info_static(), []([[maybe_unused]] const std::shared_ptr<ov::Node>&) {
                    return std::make_shared<InferType>();
                }};
    }

    // Helper to register an operation with ShapeInfer nested in the operation
    template <typename Op>
    static IShapeInferSnippetsFactory::TRegistry::value_type make_specific() {
        return {Op::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) {
                    return std::make_shared<typename Op::ShapeInfer>(n);
                }};
    }

    // Helper to register an operation with external shape infer implementation
    template <typename Op, typename InferType>
    static IShapeInferSnippetsFactory::TRegistry::value_type make_specific_external() {
        return {Op::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) {
                    return std::make_shared<InferType>(n);
                }};
    }
};

}  // namespace ov::snippets
