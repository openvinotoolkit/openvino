// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <algorithm>

#include <transformations_visibility.hpp>

#include <ngraph/pass/pass.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertPrecision;
class TRANSFORMATIONS_API FuseShapeOfConvert;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPrecision : public ngraph::pass::FunctionPass {
public:
    ConvertPrecision(ngraph::element::Type_t from, ngraph::element::Type_t to)
        : FunctionPass(),
        m_from(from),
        m_to(to) {}

    bool run_on_function(std::shared_ptr<Function> f) override;
private:
    element::Type m_from, m_to;
};
