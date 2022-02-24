// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset6.hpp>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeLoop : public ngraph::opset6::Loop {
public:
    OPENVINO_OP("StaticShapeLoop", "VPUOpset");

    explicit StaticShapeLoop(const Loop& loop);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor&) override;

protected:
    ngraph::PartialShape m_evaluatedIterationsCount;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
