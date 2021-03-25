// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset6.hpp>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeLoop : public ngraph::opset6::Loop {
public:
    static constexpr NodeTypeInfo type_info{"StaticShapeLoop", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    explicit StaticShapeLoop(const Loop& loop);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor&) override;

protected:
    ngraph::PartialShape m_evaluatedIterationsCount;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
