// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/node.hpp"
#include <ngraph/opsets/opset3.hpp>

#include <memory>
#include <vector>

namespace ngraph { namespace vpu { namespace op {

class StaticShapeReshape : public ngraph::opset3::Reshape {
public:
    OPENVINO_OP("StaticShapeReshape", "VPUOpset");

    StaticShapeReshape(const Output<Node>& arg, const Output<Node>& pattern, bool special_zero);
    explicit StaticShapeReshape(const std::shared_ptr<ngraph::opset3::Reshape>& reshape);

    void validate_and_infer_types() override;

protected:
    ngraph::PartialShape m_evaluatedOutputShape;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
