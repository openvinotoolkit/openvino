// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset6.hpp>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeLoop : public ngraph::opset6::Loop {
public:
    NGRAPH_RTTI_DECLARATION;

    explicit StaticShapeLoop(const Loop& loop);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor&) override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
