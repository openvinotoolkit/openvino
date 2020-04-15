// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

#include <memory>

namespace ngraph { namespace op {

class DynamicShapeResolver : public Op {
public:
    static constexpr NodeTypeInfo type_info{"DynamicShapeResolver", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    DynamicShapeResolver(const Output<Node>& tensorWithData, const Output<Node>& tensorWithDims);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

}  // namespace op
}  // namespace ngraph
