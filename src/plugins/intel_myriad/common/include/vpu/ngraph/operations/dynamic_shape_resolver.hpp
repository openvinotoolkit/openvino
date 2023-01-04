// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"

#include <memory>

namespace ngraph { namespace vpu { namespace op {

enum class DynamicShapeResolverMode {
    INFER_UPPER_BOUND_SHAPE,
    INFER_DYNAMIC_SHAPE
};

class DynamicShapeResolver : public ngraph::op::Op {
public:
    OPENVINO_OP("DynamicShapeResolver", "VPUOpset");

    DynamicShapeResolver(const Output<Node>& tensorWithData,
                         const Output<Node>& tensorWithDims,
                         const DynamicShapeResolverMode& mode = DynamicShapeResolverMode::INFER_UPPER_BOUND_SHAPE,
                         const ngraph::PartialShape& output_partial_shape = ngraph::PartialShape{});

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END

    // Deprecated. Left for compatibility with tests
    void setMode(DynamicShapeResolverMode mode) { m_mode = mode; }
    DynamicShapeResolverMode getMode() const { return m_mode; }

    void setOutputPartialShape(const ngraph::PartialShape& output_partial_shape) { m_output_partial_shape = output_partial_shape; }
    const ngraph::PartialShape& getOutputPartialShape() const { return m_output_partial_shape; }

private:
    DynamicShapeResolverMode m_mode;
    ngraph::PartialShape m_output_partial_shape;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
