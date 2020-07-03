// Copyright (C) 2020 Intel Corporation
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
    StaticShapeReshape(const Output<Node>& arg, const Output<Node>& pattern, bool special_zero);
    explicit StaticShapeReshape(const std::shared_ptr<ngraph::opset3::Reshape>& reshape);

    static constexpr NodeTypeInfo type_info{"StaticShapeReshape", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    void validate_and_infer_types() override;
};

}  // namespace op
}  // namespace vpu
}  // namespace ngraph
