// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/pass.hpp>
#include <low_precision/lpt_visibility.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupAvgPoolPrecisionPreserved;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    MarkupAvgPoolPrecisionPreserved();
};
