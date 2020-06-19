// Copyright (C) 2020 Intel Corporation
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
    static constexpr NodeTypeInfo type_info{"DynamicShapeResolver", 0};

    const NodeTypeInfo& get_type_info() const override { return type_info; }

    DynamicShapeResolver(const Output<Node>& tensorWithData,
                         const Output<Node>& tensorWithDims,
                         const DynamicShapeResolverMode& mode = DynamicShapeResolverMode::INFER_UPPER_BOUND_SHAPE);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) override;

    void setMode(DynamicShapeResolverMode mode) { m_mode = mode; }
    DynamicShapeResolverMode getMode() { return m_mode; }

private:
    DynamicShapeResolverMode m_mode;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
