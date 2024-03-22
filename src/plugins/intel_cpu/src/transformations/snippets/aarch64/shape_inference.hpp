// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <snippets/shape_inference/shape_inference.hpp>

namespace ov {
namespace snippets {

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
    ShapeInferPtr get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key, const std::shared_ptr<ov::Node>& op) const override;
};

} // namespace snippets
} // namespace ov
