// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertConvertSaturationAfterInputs
 * @brief Inserts ConvertSaturation op after Parameters and Scalars to convert data type of inputs
 *        to supported exexution data type.
 * @ingroup snippets
 */
class InsertConvertSaturationAfterInputs: public ngraph::pass::MatcherPass {
public:
    InsertConvertSaturationAfterInputs(const ov::element::Type exec_type = ov::element::f32);
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
